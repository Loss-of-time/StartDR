import math
from collections.abc import Callable

from ...schema.drugrec import DrugRecMedicine
from ...schema.drugrec_task import DrugRecCase
from ...schema.model.gnn_reranker import (
    DrugNodeNumericFeature,
    GNNCandidateTarget,
    GNNEdge,
    GNNEdgeType,
    GNNGraphSample,
    GNNNode,
    GNNNodeType,
    NumericFeatureStats,
)

###############################################################
# 对外公开函数
###############################################################


def fit_numeric_feature_stats(
    cases: list[DrugRecCase],
) -> NumericFeatureStats:
    """拟合候选药数值特征的标准化统计量。"""
    score_logs: list[float] = []
    for case in cases:
        for candidate in case["candidate_drugs"]:
            if candidate["score"] is None:
                continue
            score_logs.append(math.log1p(candidate["score"]))
    if not score_logs:
        return {
            "score_log_mean": 0.0,
            "score_log_std": 1.0,
        }
    score_log_mean = sum(score_logs) / len(score_logs)
    variance = sum(
        (score_log - score_log_mean) ** 2 for score_log in score_logs
    ) / len(score_logs)
    score_log_std = math.sqrt(variance)
    return {
        "score_log_mean": score_log_mean,
        "score_log_std": 1.0 if score_log_std == 0.0 else score_log_std,
    }


###############################################################
# 业务私有函数
###############################################################


class GNNGraphSampleBuilder:
    def __init__(self, case: DrugRecCase) -> None:
        """接收单个病例并准备构图上下文。"""
        self.case = case
        self.patient = case["patient"]
        on_medicine_names = [
            medicine["name"] for medicine in self.patient["on_medicine"]
        ]
        self.patient_text = "\n".join(
            [
                f"年龄：{self.patient['age']}",
                f"性别：{self.patient['gender']}",
                f"分组：{_join_text(self.patient['group'])}",
                f"诊断：{_join_text(self.patient['diagnosis'])}",
                f"症状：{_join_text(self.patient['symptom'])}",
                f"既往史：{_join_text(self.patient['antecedents'])}",
                f"过敏原：{_join_text(self.patient['allergen'])}",
                f"当前用药：{_join_text(on_medicine_names)}",
            ]
        )
        self.candidate_by_drugid = {
            candidate["drugid"]: candidate
            for candidate in case["candidate_drugs"]
        }
        self.on_medicine_by_drugid = {
            medicine["drugid"]: medicine
            for medicine in self.patient["on_medicine"]
        }
        self.nodes_by_id: dict[str, GNNNode] = {}
        self.edges: list[GNNEdge] = []
        self.edge_keys: set[tuple[str, str, str]] = set()
        self.drug_numeric_features: dict[str, DrugNodeNumericFeature] = {}
        self.candidate_targets: list[GNNCandidateTarget] = []

    def build(self) -> GNNGraphSample:
        """构造 GNN 使用的局部图样本。"""
        ordered_medicines: list[DrugRecMedicine] = []
        seen_drugids: set[str] = set()
        for candidate in self.case["candidate_drugs"]:
            drugid = candidate["drugid"]
            if drugid in seen_drugids:
                continue
            ordered_medicines.append(candidate["drug"])
            seen_drugids.add(drugid)
        for medicine in self.patient["on_medicine"]:
            drugid = medicine["drugid"]
            if drugid in seen_drugids:
                continue
            ordered_medicines.append(medicine)
            seen_drugids.add(drugid)

        for medicine in ordered_medicines:
            self._add_medicine(medicine)
        self.candidate_targets.extend(
            [
                GNNCandidateTarget(
                    drug_node_id=f"drug:{candidate['drugid']}",
                    drugid=candidate["drugid"],
                    label=1 if candidate["is_gold"] else 0,
                )
                for candidate in self.case["candidate_drugs"]
            ]
        )
        return GNNGraphSample(
            patient_id=self.case["patient_id"],
            split=self.case["split"],
            patient_text=self.patient_text,
            gold_drugids=list(self.case["gold_drugids"]),
            nodes=list(self.nodes_by_id.values()),
            edges=self.edges,
            drug_numeric_features=self.drug_numeric_features,
            candidate_targets=self.candidate_targets,
        )

    def _add_medicine(self, medicine: DrugRecMedicine) -> None:
        """把单个药物及其属性节点写入图样本。"""
        drugid = medicine["drugid"]
        drug_node_id = f"drug:{drugid}"
        self.nodes_by_id.setdefault(
            drug_node_id,
            GNNNode(
                node_id=drug_node_id,
                node_type="drug",
                text=_build_drug_node_text(medicine),
            ),
        )
        candidate = self.candidate_by_drugid.get(drugid)
        if candidate is None:
            retrieval_score = None
            retrieval_rank = None
            is_candidate = 0
        else:
            retrieval_score = candidate["score"]
            retrieval_rank = candidate["rank"]
            is_candidate = 1
        self.drug_numeric_features[drug_node_id] = DrugNodeNumericFeature(
            retrieval_score=retrieval_score,
            retrieval_rank=retrieval_rank,
            is_candidate=is_candidate,
            is_on_medicine=1 if drugid in self.on_medicine_by_drugid else 0,
        )
        self._add_attribute_nodes(
            drug_node_id=drug_node_id,
            items=medicine["treat"],
            node_type="treat",
            forward_edge_type="drug_has_treat",
            reverse_edge_type="rev_drug_has_treat",
            get_text=lambda treat: treat["treat"] or "",
            get_node_id=lambda treat: (
                f"treat:id:{treat['treat_id']}"
                if treat["treat_id"] is not None
                else f"treat:text:{(treat['treat'] or '').strip()}"
            ),
        )
        self._add_attribute_nodes(
            drug_node_id=drug_node_id,
            items=medicine["caution"],
            node_type="caution",
            forward_edge_type="drug_has_caution",
            reverse_edge_type="rev_drug_has_caution",
            get_text=lambda caution: _build_caution_text(
                caution["crowd"],
                caution["caution_level"],
            ),
            get_node_id=lambda caution: (
                f"caution:id:{caution['crowd_id']}:{caution['caution_levelid']}"
                if caution["caution_levelid"] is not None
                else "caution:text:"
                f"{caution['crowd'].strip()}:"
                f"{(caution['caution_level'] or '').strip()}"
            ),
        )
        self._add_attribute_nodes(
            drug_node_id=drug_node_id,
            items=medicine["ingredients"],
            node_type="ingredient",
            forward_edge_type="drug_has_ingredient",
            reverse_edge_type="rev_drug_has_ingredient",
            get_text=lambda ingredient: ingredient["ingredient"] or "",
            get_node_id=lambda ingredient: (
                f"ingredient:id:{ingredient['ingredient_id']}"
                if ingredient["ingredient_id"] is not None
                else f"ingredient:text:{(ingredient['ingredient'] or '').strip()}"
            ),
        )
        self._add_attribute_nodes(
            drug_node_id=drug_node_id,
            items=medicine["interaction"],
            node_type="interaction",
            forward_edge_type="drug_has_interaction",
            reverse_edge_type="rev_drug_has_interaction",
            get_text=lambda interaction: interaction["name"],
            get_node_id=lambda interaction: (
                f"interaction:id:{interaction['interaction_id']}:"
                f"{interaction['name'].strip()}"
            ),
        )

    def _add_attribute_nodes[T](
        self,
        drug_node_id: str,
        items: list[T],
        node_type: GNNNodeType,
        forward_edge_type: GNNEdgeType,
        reverse_edge_type: GNNEdgeType,
        get_text: Callable[[T], str],
        get_node_id: Callable[[T], str],
    ) -> None:
        """为药物追加指定类型的属性节点。"""
        for item in items:
            text = get_text(item).strip()
            if not text:
                continue
            node_id = get_node_id(item)
            self.nodes_by_id.setdefault(
                node_id,
                GNNNode(node_id=node_id, node_type=node_type, text=text),
            )
            self._add_bidirectional_edge(
                forward_edge_type,
                reverse_edge_type,
                drug_node_id,
                node_id,
            )

    def _add_bidirectional_edge(
        self,
        forward_edge_type: GNNEdgeType,
        reverse_edge_type: GNNEdgeType,
        drug_node_id: str,
        attr_node_id: str,
    ) -> None:
        """在药物节点和属性节点之间补双向边。"""
        forward_key = (forward_edge_type, drug_node_id, attr_node_id)
        if forward_key not in self.edge_keys:
            self.edges.append(
                GNNEdge(
                    edge_type=forward_edge_type,
                    src_node_id=drug_node_id,
                    dst_node_id=attr_node_id,
                )
            )
            self.edge_keys.add(forward_key)

        reverse_key = (reverse_edge_type, attr_node_id, drug_node_id)
        if reverse_key not in self.edge_keys:
            self.edges.append(
                GNNEdge(
                    edge_type=reverse_edge_type,
                    src_node_id=attr_node_id,
                    dst_node_id=drug_node_id,
                )
            )
            self.edge_keys.add(reverse_key)


def _build_drug_node_text(medicine: DrugRecMedicine) -> str:
    """整理药物节点的展示文本。"""
    treat_text = _join_text(
        [treat["treat"] or "" for treat in medicine["treat"]]
    )
    caution_text = _join_text(
        [
            _build_caution_text(caution["crowd"], caution["caution_level"])
            for caution in medicine["caution"]
        ]
    )
    ingredient_text = _join_text(
        [
            ingredient["ingredient"] or ""
            for ingredient in medicine["ingredients"]
        ]
    )
    interaction_text = _join_text(
        [interaction["name"] for interaction in medicine["interaction"]]
    )
    return "\n".join(
        [
            f"药名：{medicine['name']}",
            f"治疗：{treat_text}",
            f"禁用：{caution_text}",
            f"成分：{ingredient_text}",
            f"相互作用：{interaction_text}",
        ]
    )


def _build_caution_text(crowd: str, caution_level: str | None) -> str:
    """整理慎用信息的单条文本。"""
    parts = [crowd.strip()]
    if caution_level and caution_level.strip():
        parts.append(caution_level.strip())
    return " ".join(parts).strip()


###############################################################
# 通用工具函数
###############################################################


def _join_text(items: list[str]) -> str:
    """把字符串列表拼成统一展示文本。"""
    values = [item.strip() for item in items if item.strip()]
    return "、".join(values) if values else "无"
