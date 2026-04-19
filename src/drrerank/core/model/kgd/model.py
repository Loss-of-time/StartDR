"""KGDNet 最短路径复刻实现。"""

import math
from collections.abc import Sequence
from typing import cast

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, RGCNConv

from .schema import KGDForwardResult, KGDModelConfig

type HiddenEmbeddings = Float[Tensor, "batch hidden"]
type NodeIds = Int[Tensor, "node"]
type EdgeIndex = Int[Tensor, "two edge"]
type EdgeTypes = Int[Tensor, "edge"]
type EdgeWeights = Float[Tensor, "edge"]


def build_graph_node_ids(
    graph: Data,
    minimum_node_count: int,
    device: torch.device,
) -> NodeIds:
    """恢复图节点编号。"""

    if graph.x is not None and graph.x.numel() > 0:
        graph_node_ids: NodeIds = graph.x.reshape(-1).to(device=device, dtype=torch.int64)
    else:
        graph_node_ids = torch.empty((0,), dtype=torch.int64, device=device)

    node_count_from_x: int = (
        int(graph_node_ids.max().item()) + 1 if graph_node_ids.numel() > 0 else 0
    )
    edge_index: EdgeIndex = graph.edge_index.to(device=device, dtype=torch.int64)
    node_count_from_edges: int = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
    required_node_count: int = max(minimum_node_count, node_count_from_x, node_count_from_edges)

    expected_ids: NodeIds = torch.arange(required_node_count, dtype=torch.int64, device=device)
    if graph_node_ids.numel() == required_node_count and torch.equal(graph_node_ids, expected_ids):
        return graph_node_ids
    # 目的：运行时 DDI 图当前少给了最后一个节点，这里统一补成连续编号。
    return expected_ids


def build_edge_index(
    graph: Data,
    device: torch.device,
) -> EdgeIndex:
    """读取 edge_index。"""

    return graph.edge_index.to(device=device, dtype=torch.int64)


def build_edge_types(
    graph: Data,
    edge_count: int,
    device: torch.device,
) -> EdgeTypes:
    """读取 relation id。"""

    if graph.edge_type is None or graph.edge_type.numel() == 0:
        return torch.zeros((edge_count,), dtype=torch.int64, device=device)
    return graph.edge_type.to(device=device, dtype=torch.int64)


def build_edge_weights(
    graph: Data,
    edge_count: int,
    device: torch.device,
) -> EdgeWeights | None:
    """读取边权重。"""

    if not hasattr(graph, "edge_weights"):
        return None
    graph_edge_weights: Tensor | None = getattr(graph, "edge_weights")
    if graph_edge_weights is None or graph_edge_weights.numel() == 0:
        return None
    edge_weights: EdgeWeights = graph_edge_weights.to(device=device, dtype=torch.float32)
    if edge_weights.size(0) == edge_count:
        return edge_weights
    return None


class ClinicalGraphEncoder(nn.Module):
    """编码 admission 级临床图。"""

    def __init__(
        self,
        node_count: int,
        embed_dim: int,
        relation_count: int,
        dropout: float,
    ) -> None:
        """初始化临床图编码器。

        Args:
            node_count: 临床图节点总数。
            embed_dim: 嵌入维度。
            relation_count: 临床 relation 数量。
            dropout: dropout 比例。
        """

        super().__init__()
        self.node_count = node_count
        self.embedding = nn.Embedding(node_count, embed_dim)
        self.gnn1 = RGCNConv(embed_dim, embed_dim, num_relations=relation_count)
        self.gnn2 = RGCNConv(embed_dim, embed_dim, num_relations=relation_count)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化参数。"""

        nn.init.xavier_uniform_(self.embedding.weight)
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()

    def forward(self, graph: Data) -> HiddenEmbeddings:
        """执行临床图编码。"""

        device: torch.device = self.embedding.weight.device
        node_ids: NodeIds = build_graph_node_ids(graph, self.node_count, device)
        edge_index: EdgeIndex = build_edge_index(graph, device)
        edge_types: EdgeTypes = build_edge_types(graph, edge_index.size(1), device)

        hidden: HiddenEmbeddings = self.embedding(node_ids)
        hidden = self.gnn1(hidden, edge_index, edge_types)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.gnn2(hidden, edge_index, edge_types)
        hidden = F.relu(hidden)
        return hidden.mean(dim=0, keepdim=True)


class MedicineGraphEncoder(nn.Module):
    """编码 admission 级药物图或全局 DDI 图。"""

    def __init__(
        self,
        node_count: int,
        embed_dim: int,
        dropout: float,
    ) -> None:
        """初始化药物图编码器。

        Args:
            node_count: 药物图节点总数。
            embed_dim: 嵌入维度。
            dropout: dropout 比例。
        """

        super().__init__()
        self.node_count = node_count
        self.embedding = nn.Embedding(node_count, embed_dim)
        self.gnn1 = GCNConv(embed_dim, embed_dim)
        self.gnn2 = GCNConv(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化参数。"""

        nn.init.xavier_uniform_(self.embedding.weight)
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()

    def forward(self, graph: Data) -> HiddenEmbeddings:
        """执行药物图编码。"""

        device: torch.device = self.embedding.weight.device
        node_ids: NodeIds = build_graph_node_ids(graph, self.node_count, device)
        edge_index: EdgeIndex = build_edge_index(graph, device)
        edge_weights: EdgeWeights | None = build_edge_weights(graph, edge_index.size(1), device)

        hidden: HiddenEmbeddings = self.embedding(node_ids)
        hidden = self.gnn1(hidden, edge_index, edge_weights)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.gnn2(hidden, edge_index, edge_weights)
        hidden = F.relu(hidden)
        return hidden.mean(dim=0, keepdim=True)


class AdmissionSequenceEncoder(nn.Module):
    """编码 admission 序列。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """初始化 admission 序列编码器。

        Args:
            input_dim: 输入维度。
            hidden_dim: 隐状态维度。
            num_layers: GRU 层数。
            dropout: dropout 比例。
        """

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化参数。"""

        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Float[Tensor, "layer batch hidden"]:
        """构造初始隐状态。"""

        return torch.zeros(
            (self.num_layers, batch_size, self.hidden_dim),
            dtype=torch.float32,
            device=device,
        )

    def forward(self, embeddings: Sequence[HiddenEmbeddings]) -> Float[Tensor, "step hidden"]:
        """逐步编码 admission 序列。"""

        # 目的：通过已注册参数获取设备，避免静态检查将模块属性推断为宽泛联合类型。
        gru_parameter: Tensor = next(self.gru.parameters())
        device: torch.device = gru_parameter.device
        hidden_state: Float[Tensor, "layer batch hidden"] = self.init_hidden(1, device)
        step_outputs: list[HiddenEmbeddings] = []

        for embedding in embeddings:
            gru_input: Float[Tensor, "batch seq hidden"] = embedding.to(device).unsqueeze(1)
            _, hidden_state = self.gru(gru_input, hidden_state)
            # 目的：保留原始实现按层汇聚隐藏状态的时序表示。
            step_outputs.append(hidden_state.sum(dim=0))

        return torch.cat(step_outputs, dim=0)


class FusionConv(nn.Module):
    """融合临床与药物两路特征。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        """初始化融合卷积层。

        Args:
            input_dim: 输入维度。
            hidden_dim: 隐层维度。
            output_dim: 输出维度。
            num_layers: 中间层数量。
        """

        super().__init__()
        self.initial_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        # 目的：显式收窄 ModuleList 元素类型，避免静态检查把 layer 推断成宽泛 Module。
        hidden_layers: list[nn.Conv1d] = [
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1) for _ in range(num_layers)
        ]
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.final_conv = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.activation = nn.ReLU()
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化参数。"""

        nn.init.xavier_uniform_(self.initial_conv.weight)
        if self.initial_conv.bias is not None:
            nn.init.zeros_(self.initial_conv.bias)
        for hidden_layer in self.hidden_layers:
            # 目的：逐层收窄为 Conv1d，消除 ModuleList 迭代时的宽泛类型。
            layer: nn.Conv1d = cast(nn.Conv1d, hidden_layer)
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, embedding: HiddenEmbeddings) -> HiddenEmbeddings:
        """执行双路特征融合。"""

        hidden: Float[Tensor, "batch channel length"] = embedding.unsqueeze(-1)
        hidden = self.activation(self.initial_conv(hidden))
        for hidden_layer in self.hidden_layers:
            # 目的：在前向阶段保持卷积层的明确静态类型。
            layer: nn.Conv1d = cast(nn.Conv1d, hidden_layer)
            hidden = self.activation(layer(hidden))
        hidden = self.final_conv(hidden)
        return self.pooling(hidden).squeeze(-1)


class FusionMLP(nn.Module):
    """把融合后的特征投影到 joint 序列空间。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        """初始化融合 MLP。

        Args:
            input_dim: 输入维度。
            hidden_dim: 隐层维度。
            output_dim: 基础输出维度。
        """

        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim * 2)
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化参数。"""

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, embedding: HiddenEmbeddings) -> HiddenEmbeddings:
        """执行 joint 特征投影。"""

        hidden: HiddenEmbeddings = self.layer1(embedding)
        hidden = self.activation(hidden)
        return self.layer2(hidden)


class PredictionMLP(nn.Module):
    """输出药物打分。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        """初始化预测头。

        Args:
            input_dim: 输入维度。
            hidden_dim: 隐层维度。
            output_dim: 输出维度。
        """

        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化参数。"""

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, embedding: HiddenEmbeddings) -> HiddenEmbeddings:
        """执行预测。"""

        hidden: HiddenEmbeddings = self.layer1(embedding)
        hidden = self.activation(hidden)
        hidden = self.layer2(hidden)
        return self.output_activation(hidden)


class AdmissionAttention(nn.Module):
    """用 clinical-query 与 joint-memory 生成药物打分。"""

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        """初始化注意力预测层。

        Args:
            embed_dim: 注意力隐空间维度。
            output_dim: 药物打分维度。
            attention_heads: 多头数。
            dropout: dropout 比例。
        """

        super().__init__()
        head_count: int = math.gcd(embed_dim, attention_heads)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=head_count,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.predictor = PredictionMLP(embed_dim, embed_dim * 2, output_dim)

    def forward(
        self,
        clinical_context: HiddenEmbeddings,
        joint_context: HiddenEmbeddings,
        residual_context: HiddenEmbeddings,
    ) -> HiddenEmbeddings:
        """执行注意力预测。"""

        query: Float[Tensor, "batch seq hidden"] = clinical_context.unsqueeze(1)
        key: Float[Tensor, "batch seq hidden"] = joint_context.unsqueeze(1)
        value: Float[Tensor, "batch seq hidden"] = joint_context.unsqueeze(1)
        attention_output: Float[Tensor, "batch seq hidden"]
        attention_output, _ = self.attention(query, key, value, need_weights=False)
        normalized_output: Float[Tensor, "batch seq hidden"] = self.layer_norm(
            residual_context.unsqueeze(1) + attention_output,
        )
        return self.predictor(normalized_output.squeeze(1))


class KGDNet(nn.Module):
    """当前工程可直接接入 runtime 输出的 KGDNet 复刻版。"""

    def __init__(
        self,
        config: KGDModelConfig,
    ) -> None:
        """初始化模型。

        Args:
            config: 模型配置。
        """

        super().__init__()
        self.config = config
        self.clinical_encoder = ClinicalGraphEncoder(
            node_count=config.clinical_node_count,
            embed_dim=config.embed_dim,
            relation_count=config.clinical_relations,
            dropout=config.dropout,
        )
        self.medicine_encoder = MedicineGraphEncoder(
            node_count=config.medicine_node_count,
            embed_dim=config.embed_dim,
            dropout=config.dropout,
        )
        self.ddi_encoder = MedicineGraphEncoder(
            node_count=config.medicine_node_count,
            embed_dim=config.embed_dim,
            dropout=config.dropout,
        )
        self.clinical_sequence_encoder = AdmissionSequenceEncoder(
            input_dim=config.embed_dim,
            hidden_dim=config.embed_dim,
            num_layers=config.emb_rnn_layers,
            dropout=config.dropout,
        )
        self.medicine_sequence_encoder = AdmissionSequenceEncoder(
            input_dim=config.embed_dim,
            hidden_dim=config.embed_dim,
            num_layers=config.emb_rnn_layers,
            dropout=config.dropout,
        )
        self.fusion_conv = FusionConv(
            input_dim=config.embed_dim * 2,
            hidden_dim=config.embed_dim,
            output_dim=config.embed_dim,
            num_layers=config.fusion_layers,
        )
        self.fusion_mlp = FusionMLP(
            input_dim=config.embed_dim,
            hidden_dim=config.embed_dim * 2,
            output_dim=config.embed_dim,
        )
        self.joint_sequence_encoder = AdmissionSequenceEncoder(
            input_dim=config.embed_dim * 2,
            hidden_dim=config.embed_dim,
            num_layers=config.joint_rnn_layers,
            dropout=config.dropout,
        )
        self.prediction_layer = AdmissionAttention(
            embed_dim=config.embed_dim,
            output_dim=config.medicine_node_count,
            attention_heads=config.attention_heads,
            dropout=config.dropout,
        )

    def forward(
        self,
        patient_graphs: list[list[Data]],
        ddi_graph: Data,
    ) -> KGDForwardResult:
        """对单个病人的 admission 序列执行前向。

        Args:
            patient_graphs: 单个病人的 admission 图序列，每个元素依次为临床图和药物图。
            ddi_graph: 全局 DDI 图。

        Returns:
            KGDForwardResult: admission 级药物打分与中间时序表征。
        """

        ddi_embedding: HiddenEmbeddings = self.ddi_encoder(ddi_graph)
        clinical_graph_embeddings: list[HiddenEmbeddings] = []
        medicine_graph_embeddings: list[HiddenEmbeddings] = []

        for admission_graphs in patient_graphs:
            clinical_graph: Data = admission_graphs[0]
            medicine_graph: Data = admission_graphs[1]
            clinical_graph_embeddings.append(self.clinical_encoder(clinical_graph))
            # 目的：保留原始 KGDNet 用 DDI 图表示抵消药物图表示的主设计。
            medicine_graph_embeddings.append(self.medicine_encoder(medicine_graph) - ddi_embedding)

        clinical_sequence: Float[Tensor, "admission hidden"] = self.clinical_sequence_encoder(
            clinical_graph_embeddings,
        )
        medicine_sequence: Float[Tensor, "admission hidden"] = self.medicine_sequence_encoder(
            medicine_graph_embeddings,
        )

        fusion_inputs: list[HiddenEmbeddings] = []
        for admission_index in range(len(patient_graphs)):
            fusion_inputs.append(
                torch.cat(
                    (
                        clinical_sequence[admission_index : admission_index + 1],
                        medicine_sequence[admission_index : admission_index + 1],
                    ),
                    dim=1,
                )
            )

        joint_inputs: list[HiddenEmbeddings] = []
        for fusion_input in fusion_inputs:
            joint_inputs.append(self.fusion_mlp(self.fusion_conv(fusion_input)))

        joint_sequence: Float[Tensor, "admission hidden"] = self.joint_sequence_encoder(
            joint_inputs
        )

        logits_per_admission: list[HiddenEmbeddings] = []
        for admission_index in range(len(patient_graphs)):
            logits_per_admission.append(
                self.prediction_layer(
                    clinical_context=clinical_sequence[admission_index : admission_index + 1],
                    joint_context=joint_sequence[admission_index : admission_index + 1],
                    residual_context=clinical_graph_embeddings[admission_index],
                )
            )

        logits: Float[Tensor, "admission medicine"] = torch.cat(logits_per_admission, dim=0)
        probabilities: Float[Tensor, "admission medicine"] = torch.sigmoid(logits)
        return KGDForwardResult(
            logits=logits,
            probabilities=probabilities,
            clinical_sequence=clinical_sequence,
            medicine_sequence=medicine_sequence,
            joint_sequence=joint_sequence,
        )
