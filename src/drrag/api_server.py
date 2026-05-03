"""在线推荐推理 API 入口。"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .core.inference_service import (
    DrugSearchHit,
    OnlineInferenceConfig,
    OnlineInferenceService,
    OnlinePatientPayload,
)
from .core.setting import (
    DEFAULT_API_ALLOWED_ORIGINS,
    DEFAULT_API_CHECKPOINT_PATH,
    DEFAULT_API_DISPLAY_TOP_K,
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_API_RETRIEVAL_TOP_K,
    DEFAULT_RAG_MAX_EVIDENCES_PER_CANDIDATE,
    DEFAULT_SILICONFLOW_MODEL,
)


class RecommendRequestModel(BaseModel):
    """推荐接口请求体。"""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., description="患者年龄。")
    gender: str = Field(..., description="患者性别。")
    group: list[str] = Field(..., description="患者分组。")
    diagnosis: list[str] = Field(..., description="诊断列表。")
    symptom: list[str] = Field(..., description="症状列表。")
    antecedents: list[str] = Field(..., description="既往史列表。")
    allergen: list[str] = Field(..., description="过敏史列表。")
    on_medicine_drugids: list[str] = Field(..., description="当前用药 drugid 列表。")

    def to_payload(self) -> OnlinePatientPayload:
        """转换为服务层输入对象。"""

        payload: OnlinePatientPayload = OnlinePatientPayload(
            age=self.age,
            gender=self.gender,
            group=list(self.group),
            diagnosis=list(self.diagnosis),
            symptom=list(self.symptom),
            antecedents=list(self.antecedents),
            allergen=list(self.allergen),
            on_medicine_drugids=list(self.on_medicine_drugids),
        )
        return payload


def create_app(
    service: OnlineInferenceService,
    allowed_origins: Sequence[str],
) -> FastAPI:
    """构造 FastAPI 应用。

    Args:
        service: 在线推理服务。
        allowed_origins: 允许跨域的前端来源。

    Returns:
        已挂载路由的应用实例。
    """

    app: FastAPI = FastAPI(
        title="StartDR Online Recommendation API",
        description="StartDR 最小在线推荐推理 API。",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        """返回服务健康状态。"""

        payload: dict[str, object] = service.build_health_payload()
        return payload

    @app.get("/api/drugs/search")
    def search_drugs(
        q: Annotated[str, Query(description="药名或批准文号关键字。")],
        limit: Annotated[int, Query(ge=1, le=50)] = 20,
    ) -> dict[str, object]:
        """按药名或批准文号搜索药品。"""

        hits: list[DrugSearchHit] = service.search_drugs(q, limit)
        payload: dict[str, object] = {
            "query": q,
            "limit": limit,
            "hits": [
                {
                    "drugid": hit.drugid,
                    "name": hit.name,
                    "CMAN": hit.CMAN,
                    "treat_summary": hit.treat_summary,
                    "same_name_count": hit.same_name_count,
                }
                for hit in hits
            ],
        }
        return payload

    @app.post("/api/recommend")
    def recommend(request: RecommendRequestModel) -> dict[str, object]:
        """执行在线推荐。"""

        try:
            response = service.recommend(request.to_payload())
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return response.to_dict()

    return app


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="启动 StartDR 在线推荐推理 API。"
    )
    parser.add_argument("--host", type=str, default=DEFAULT_API_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_API_CHECKPOINT_PATH))
    parser.add_argument("--rag-model", type=str, default=DEFAULT_SILICONFLOW_MODEL)
    parser.add_argument("--retrieval-top-k", type=int, default=DEFAULT_API_RETRIEVAL_TOP_K)
    parser.add_argument("--display-top-k", type=int, default=DEFAULT_API_DISPLAY_TOP_K)
    parser.add_argument(
        "--max-evidences-per-candidate",
        type=int,
        default=DEFAULT_RAG_MAX_EVIDENCES_PER_CANDIDATE,
    )
    parser.add_argument(
        "--cors-origins",
        type=str,
        default=",".join(DEFAULT_API_ALLOWED_ORIGINS),
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    allowed_origins: tuple[str, ...] = tuple(
        origin.strip() for origin in args.cors_origins.split(",") if origin.strip() != ""
    )
    config: OnlineInferenceConfig = OnlineInferenceConfig(
        checkpoint_path=Path(args.checkpoint),
        host=args.host,
        port=args.port,
        allowed_origins=allowed_origins,
        retrieval_top_k=args.retrieval_top_k,
        display_top_k=args.display_top_k,
        rag_model_name=args.rag_model,
        max_evidences_per_candidate=args.max_evidences_per_candidate,
    )
    service: OnlineInferenceService = OnlineInferenceService.build(config)
    app: FastAPI = create_app(service, config.allowed_origins)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
