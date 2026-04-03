from ..schema.drugrec_task import DrugRecCase, DrugRecModelName
from .drugrec_model import DrugRecModel
from .gnn_reranker.model import GNNModel

MODEL_REGISTRY: dict[DrugRecModelName, type[DrugRecModel]] = {
    "gnn": GNNModel,
}


def get_model_names() -> list[DrugRecModelName]:
    """返回当前已注册的推荐模型名称。"""
    return list(MODEL_REGISTRY)


def build_model(
    name: DrugRecModelName,
    train_cases: list[DrugRecCase],
    top_k: int,
) -> DrugRecModel:
    """根据名称构建训练态推荐模型。"""
    return MODEL_REGISTRY[name].build_for_train(
        train_cases=train_cases,
        top_k=top_k,
    )
