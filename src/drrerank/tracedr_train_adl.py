"""ADL 云 GPU 上的 TraceDR 训练入口。"""

import argparse
import json
import platform
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch

from .core.model.experiment.runner import run_training_experiment
from .core.model.experiment.schema import ExperimentEvalResult, ExperimentReport, TrainingRunResult
from .core.model.tracedr.train import TraceDRTrainAdapter, TrainConfig
from .core.schema import unstructure
from .core.setting import (
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_TRACEDR_DEV_INPUT_PATH,
    DEFAULT_TRACEDR_JSONL_OUTPUT_ROOT,
    DEFAULT_TRACEDR_TRAIN_INPUT_PATH,
)

DEFAULT_TRACEDR_TEST_INPUT_PATH: Path = (
    DEFAULT_TRACEDR_JSONL_OUTPUT_ROOT / "tracedr_top50" / "test.jsonl"
)


@dataclass(slots=True)
class CommandSnapshot:
    """命令行调用快照。"""

    argv: list[str]
    command: str


@dataclass(slots=True)
class GitSnapshot:
    """Git 状态快照。"""

    commit: str
    branch: str
    status_lines: list[str]
    is_dirty: bool


@dataclass(slots=True)
class GpuSnapshot:
    """GPU 与 PyTorch 运行环境快照。"""

    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    device_count: int
    device_names: list[str]
    nvidia_smi_lines: list[str]


@dataclass(slots=True)
class RuntimeSnapshot:
    """运行时环境快照。"""

    hostname: str
    platform: str
    python_executable: str
    python_version: str
    uv_version: str
    working_directory: str
    started_at: str
    finished_at: str
    duration_seconds: float


@dataclass(slots=True)
class OutputPaths:
    """训练产物路径。"""

    detailed_report_path: str
    train_report_path: str
    checkpoint_path: str
    log_path: str | None


@dataclass(slots=True)
class MetricSnapshot:
    """核心指标摘要。"""

    best_epoch: int
    selection_metric: str
    best_metric_value: float
    best_dev_metrics: dict[str, float]
    test_metrics: dict[str, float] | None


@dataclass(slots=True)
class AdlTrainDetailedReport:
    """ADL 训练详细 JSON 报告。"""

    output_name: str
    command: CommandSnapshot
    runtime: RuntimeSnapshot
    git: GitSnapshot
    gpu: GpuSnapshot
    train_config: dict[str, object]
    output_paths: OutputPaths
    metrics: MetricSnapshot
    training_run: dict[str, object]


def build_default_detailed_report_path(output_name: str) -> Path:
    """构造详细 JSON 报告默认路径。"""

    return DEFAULT_MODEL_OUTPUT_DIR / f"{output_name}.adl.json"


def run_text_command(command: list[str]) -> str:
    """执行命令并返回标准输出文本。"""

    completed_process: subprocess.CompletedProcess[str] = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed_process.stdout.strip()


def build_git_snapshot() -> GitSnapshot:
    """采集 Git 状态。"""

    status_text: str = run_text_command(["git", "status", "--short"])
    status_lines: list[str] = [line for line in status_text.splitlines() if line]
    return GitSnapshot(
        commit=run_text_command(["git", "rev-parse", "HEAD"]),
        branch=run_text_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        status_lines=status_lines,
        is_dirty=bool(status_lines),
    )


def build_gpu_snapshot() -> GpuSnapshot:
    """采集 GPU 与 PyTorch 环境信息。"""

    cuda_available: bool = torch.cuda.is_available()
    device_count: int = torch.cuda.device_count() if cuda_available else 0
    device_names: list[str] = [
        torch.cuda.get_device_name(device_index) for device_index in range(device_count)
    ]
    nvidia_smi_text: str = run_text_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader",
        ]
    )
    nvidia_smi_lines: list[str] = [line for line in nvidia_smi_text.splitlines() if line]
    return GpuSnapshot(
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        device_count=device_count,
        device_names=device_names,
        nvidia_smi_lines=nvidia_smi_lines,
    )


def build_runtime_snapshot(
    started_at: datetime,
    finished_at: datetime,
    duration_seconds: float,
) -> RuntimeSnapshot:
    """采集运行时环境信息。"""

    return RuntimeSnapshot(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_executable=sys.executable,
        python_version=sys.version,
        uv_version=run_text_command(["uv", "--version"]),
        working_directory=str(Path.cwd()),
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        duration_seconds=duration_seconds,
    )


def serialize_train_config(config: TrainConfig) -> dict[str, object]:
    """把训练配置序列化为 JSON 友好结构。"""

    return {
        "train_input": str(config.train_input),
        "dev_input": str(config.dev_input),
        "test_input": str(config.test_input) if config.test_input is not None else None,
        "output_name": config.output_name,
        "epochs": config.epochs,
        "train_limit": config.train_limit,
        "dev_limit": config.dev_limit,
        "test_limit": config.test_limit,
        "selection_metric": config.selection_metric,
        "num_layers": config.num_layers,
        "use_evidence_supervision": config.use_evidence_supervision,
        "evidence_text_mode": config.evidence_text_mode,
        "include_on_medicine": config.include_on_medicine,
    }


def flatten_eval_result(eval_result: ExperimentEvalResult | None) -> dict[str, float] | None:
    """把评测结果拍平成单层字典。"""

    if eval_result is None:
        return None
    return eval_result.to_flat_dict()


def build_metric_snapshot(report: ExperimentReport) -> MetricSnapshot:
    """从统一实验报告中提取摘要指标。"""

    return MetricSnapshot(
        best_epoch=report.best_epoch,
        selection_metric=report.selection_metric,
        best_metric_value=report.best_metric_value,
        best_dev_metrics=report.best_dev_metrics.to_flat_dict(),
        test_metrics=flatten_eval_result(report.test_metrics),
    )


def write_detailed_report(
    report_path: Path,
    report: AdlTrainDetailedReport,
) -> None:
    """写出详细 JSON 报告。"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(unstructure(report), file, ensure_ascii=False, indent=2)


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    """把 CLI 参数转换为训练配置。"""

    return TrainConfig(
        train_input=args.train_input,
        dev_input=args.dev_input,
        test_input=args.test_input,
        output_name=args.output_name,
        epochs=args.epochs,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
        selection_metric=args.selection_metric,
        num_layers=args.num_layers,
        use_evidence_supervision=not args.disable_evidence_supervision,
        evidence_text_mode=args.evidence_text_mode,
        include_on_medicine=not args.exclude_on_medicine,
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="在 ADL 云 GPU 环境下训练 TraceDR 并产出详细 JSON 报告。"
    )
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRACEDR_TRAIN_INPUT_PATH)
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_TRACEDR_DEV_INPUT_PATH)
    parser.add_argument("--test-input", type=Path, default=DEFAULT_TRACEDR_TEST_INPUT_PATH)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--selection-metric", type=str, default="mrr")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--disable-evidence-supervision",
        action="store_true",
        help="仅保留 answer_loss，去掉 evidence_loss。",
    )
    parser.add_argument(
        "--evidence-text-mode",
        type=str,
        choices=("full", "name_only"),
        default="full",
        help="证据文本模式：full=完整属性串，name_only=仅药名。",
    )
    parser.add_argument(
        "--exclude-on-medicine",
        action="store_true",
        help="构图时移除当前在用药，仅保留 top_k_drugs。",
    )
    parser.add_argument("--detailed-report-path", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    train_config: TrainConfig = build_train_config(args)
    detailed_report_path: Path = (
        args.detailed_report_path
        if args.detailed_report_path is not None
        else build_default_detailed_report_path(train_config.output_name)
    )

    started_at: datetime = datetime.now(UTC)
    started_at_monotonic: float = time.monotonic()
    training_run: TrainingRunResult = run_training_experiment(
        train_config,
        TraceDRTrainAdapter(),
    )
    finished_at: datetime = datetime.now(UTC)
    duration_seconds: float = time.monotonic() - started_at_monotonic

    detailed_report: AdlTrainDetailedReport = AdlTrainDetailedReport(
        output_name=train_config.output_name,
        command=CommandSnapshot(
            argv=sys.argv,
            command=shlex.join(sys.argv),
        ),
        runtime=build_runtime_snapshot(
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
        ),
        git=build_git_snapshot(),
        gpu=build_gpu_snapshot(),
        train_config=serialize_train_config(train_config),
        output_paths=OutputPaths(
            detailed_report_path=str(detailed_report_path.resolve()),
            train_report_path=training_run.artifacts.report_path,
            checkpoint_path=training_run.artifacts.checkpoint_path,
            log_path=str(args.log_path.resolve()) if args.log_path is not None else None,
        ),
        metrics=build_metric_snapshot(training_run.report),
        training_run=cast(dict[str, object], unstructure(training_run)),
    )
    # 目的：把 ADL 远端训练元信息与统一实验报告合并到单一 JSON，便于后续归档与追溯。
    write_detailed_report(detailed_report_path, detailed_report)
    print(f"ADL 详细报告已写入: {detailed_report_path.resolve()}")


if __name__ == "__main__":
    main()
