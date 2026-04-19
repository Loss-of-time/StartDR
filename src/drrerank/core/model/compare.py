"""统一触发多模型训练并输出对比报告。"""

import argparse
import json
from pathlib import Path

from ..schema import unstructure
from ..setting import (
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_TRACEDR_DEV_INPUT_PATH,
    DEFAULT_TRACEDR_TRAIN_INPUT_PATH,
)
from .experiment.runner import run_training_experiment
from .experiment.schema import (
    CompareReport,
    CompareSummaryRow,
    ExperimentEvalResult,
    TrainingRunResult,
)
from .foursdrug.schema import FourSDrugTrainConfig
from .foursdrug.train import FourSDrugTrainAdapter
from .gat.train import GATTrainAdapter
from .gat.train import TrainConfig as GATTrainConfig
from .kgd.train import KGDTrainAdapter
from .kgd.train import TrainConfig as KGDTrainConfig
from .tracedr.train import TraceDRTrainAdapter
from .tracedr.train import TrainConfig as TraceDRTrainConfig

type SupportedModelName = str

SUPPORTED_MODEL_NAMES: tuple[SupportedModelName, ...] = ("tracedr", "gat", "kgd", "foursdrug")


def parse_args() -> argparse.Namespace:
    """解析统一对比实验命令行参数。

    Returns:
        解析后的参数对象。
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="统一训练多个精排模型并输出对比报告。"
    )
    parser.add_argument("--models", type=str, default="tracedr,gat,kgd,foursdrug")
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--selection-metric", type=str, default="mrr")
    parser.add_argument("--compare-metric", type=str, default="mrr")
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRACEDR_TRAIN_INPUT_PATH)
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_TRACEDR_DEV_INPUT_PATH)
    parser.add_argument("--test-input", type=Path, default=None)
    parser.add_argument("--kgd-input-dir", type=Path, default=None)
    parser.add_argument("--foursdrug-input-dir", type=Path, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--gat-encoder-model-name", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--foursdrug-batch-size", type=int, default=16)
    parser.add_argument("--foursdrug-embed-dim", type=int, default=64)
    parser.add_argument("--foursdrug-lr", type=float, default=5e-3)
    parser.add_argument("--foursdrug-alpha", type=float, default=0.5)
    parser.add_argument("--foursdrug-beta", type=float, default=1.0)
    parser.add_argument("--foursdrug-eval-threshold", type=float, default=0.8)
    return parser.parse_args()


def parse_model_names(raw_models: str) -> list[SupportedModelName]:
    """解析并校验模型名称列表。

    Args:
        raw_models: 逗号分隔的模型名称字符串。

    Returns:
        去重后的模型名称列表。
    """

    model_names: list[SupportedModelName] = []
    for raw_model_name in raw_models.split(","):
        model_name: SupportedModelName = raw_model_name.strip()
        if model_name == "":
            continue
        if model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(f"不支持的模型名称 `{model_name}`。")
        if model_name in model_names:
            continue
        model_names.append(model_name)
    if not model_names:
        raise ValueError("未提供任何可执行的模型名称。")
    return model_names


def build_tracedr_config(args: argparse.Namespace) -> TraceDRTrainConfig:
    """构造 TraceDR 对比实验配置。"""

    return TraceDRTrainConfig(
        train_input=args.train_input,
        dev_input=args.dev_input,
        test_input=args.test_input,
        output_name=f"{args.output_prefix}_tracedr",
        epochs=args.epochs,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
        selection_metric=args.selection_metric,
    )


def build_gat_config(args: argparse.Namespace) -> GATTrainConfig:
    """构造 GAT 对比实验配置。"""

    return GATTrainConfig(
        train_input=args.train_input,
        dev_input=args.dev_input,
        test_input=args.test_input,
        output_name=f"{args.output_prefix}_gat",
        epochs=args.epochs,
        encoder_model_name=args.gat_encoder_model_name,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
        selection_metric=args.selection_metric,
    )


def build_kgd_config(args: argparse.Namespace) -> KGDTrainConfig:
    """构造 KGD 对比实验配置。"""

    if args.kgd_input_dir is None:
        raise ValueError("执行 KGD 对比实验时必须提供 `--kgd-input-dir`。")
    return KGDTrainConfig(
        input_dir=args.kgd_input_dir,
        output_name=f"{args.output_prefix}_kgd",
        epochs=args.epochs,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
        selection_metric=args.selection_metric,
    )


def build_foursdrug_config(args: argparse.Namespace) -> FourSDrugTrainConfig:
    """构造 4SDrug 对比实验配置。"""

    if args.foursdrug_input_dir is None:
        raise ValueError("执行 4SDrug 对比实验时必须提供 `--foursdrug-input-dir`。")
    return FourSDrugTrainConfig(
        input_dir=args.foursdrug_input_dir,
        output_name=f"{args.output_prefix}_foursdrug",
        epochs=args.epochs,
        batch_size=args.foursdrug_batch_size,
        embed_dim=args.foursdrug_embed_dim,
        lr=args.foursdrug_lr,
        alpha=args.foursdrug_alpha,
        beta=args.foursdrug_beta,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
        eval_threshold=args.foursdrug_eval_threshold,
        selection_metric=args.selection_metric,
    )


def build_compare_row(
    compare_metric: str,
    run_result: TrainingRunResult,
) -> CompareSummaryRow:
    """把单模型训练结果转换成对比汇总行。

    Args:
        compare_metric: 对比排序指标。
        run_result: 单模型训练结果。

    Returns:
        汇总后的对比行。
    """

    target_metrics: ExperimentEvalResult
    compare_split: str
    if run_result.report.test_metrics is None:
        target_metrics = run_result.report.best_dev_metrics
        compare_split = "dev"
    else:
        target_metrics = run_result.report.test_metrics
        compare_split = "test"
    return CompareSummaryRow(
        model_name=run_result.report.experiment_name,
        output_name=run_result.report.output_name,
        best_epoch=run_result.report.best_epoch,
        selection_metric=run_result.report.selection_metric,
        best_metric_value=run_result.report.best_metric_value,
        compare_split=compare_split,
        compare_metric=compare_metric,
        compare_metric_value=target_metrics.get_metric_value(compare_metric),
        best_dev_metrics=run_result.report.best_dev_metrics.to_flat_dict(),
        test_metrics=(
            None
            if run_result.report.test_metrics is None
            else run_result.report.test_metrics.to_flat_dict()
        ),
        report_path=run_result.artifacts.report_path,
        checkpoint_path=run_result.artifacts.checkpoint_path,
    )


def run_compare(args: argparse.Namespace) -> CompareReport:
    """执行多模型对比实验。

    Args:
        args: 命令行参数。

    Returns:
        对比实验汇总报告。
    """

    model_names: list[SupportedModelName] = parse_model_names(args.models)
    rows: list[CompareSummaryRow] = []

    model_name: SupportedModelName
    for model_name in model_names:
        if model_name == "tracedr":
            run_result = run_training_experiment(build_tracedr_config(args), TraceDRTrainAdapter())
        elif model_name == "gat":
            run_result = run_training_experiment(build_gat_config(args), GATTrainAdapter())
        elif model_name == "kgd":
            run_result = run_training_experiment(build_kgd_config(args), KGDTrainAdapter())
        else:
            run_result = run_training_experiment(
                build_foursdrug_config(args), FourSDrugTrainAdapter()
            )
        rows.append(build_compare_row(args.compare_metric, run_result))

    rows.sort(
        key=lambda row: float("inf") if row.compare_metric == "loss" else row.compare_metric_value,
        reverse=True,
    )
    if args.compare_metric == "loss":
        rows.sort(key=lambda row: row.compare_metric_value)
    return CompareReport(
        output_name=f"{args.output_prefix}_compare",
        compare_metric=args.compare_metric,
        rows=rows,
    )


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    report: CompareReport = run_compare(args)
    DEFAULT_MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path: Path = DEFAULT_MODEL_OUTPUT_DIR / f"{report.output_name}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(unstructure(report), file, ensure_ascii=False, indent=2)
    print(f"对比报告已写入: {report_path.resolve()}")
