"""KGD 模型结构定义。"""

from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

type AdmissionLogits = Float[Tensor, "admission medicine"]
type AdmissionEmbeddings = Float[Tensor, "admission hidden"]


@dataclass(slots=True)
class KGDModelConfig:
    """KGDNet 复刻版模型配置。"""

    clinical_vocab_size: int
    medicine_vocab_size: int
    embed_dim: int = 64
    clinical_relations: int = 12
    emb_rnn_layers: int = 3
    fusion_layers: int = 3
    joint_rnn_layers: int = 3
    attention_heads: int = 4
    dropout: float = 0.5

    @property
    def clinical_node_count(self) -> int:
        """返回包含 patient 节点的临床节点总数。"""

        return self.clinical_vocab_size + 1

    @property
    def medicine_node_count(self) -> int:
        """返回包含 patient 占位节点的药物节点总数。"""

        return self.medicine_vocab_size + 1


@dataclass(slots=True)
class KGDForwardResult:
    """KGDNet 单个病人的前向结果。"""

    logits: AdmissionLogits
    probabilities: AdmissionLogits
    clinical_sequence: AdmissionEmbeddings
    medicine_sequence: AdmissionEmbeddings
    joint_sequence: AdmissionEmbeddings
