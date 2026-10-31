"""Every prediction passes through the same named concept bottleneck."""
# reviewed
import torch
from torch import nn


class ConceptBottleneck(nn.Module):
    def __init__(self, feature_dim: int, concepts: int, classes: int):
        super().__init__()
        self.projection = nn.Linear(feature_dim, concepts, bias=False)
        self.head = nn.Linear(concepts, classes)
        # 非参数，但需要保存的状态：state_dict() 只收集 Parameter 和 buffer，普通属性不在里面
        # 是概念标准化用的统计量，通常在训练时统计得到
        self.register_buffer("concept_mean", torch.zeros(concepts))
        self.register_buffer("concept_std", torch.ones(concepts))

    def concepts(self, features: torch.Tensor) -> torch.Tensor:
        # 这里就用到了训练时统计量来标准化
        return (self.projection(features) - self.concept_mean) / self.concept_std

    def forward(self, features: torch.Tensor):
        concepts = self.concepts(features)
        return self.head(concepts), concepts


class NoisyCBM(nn.Module):
    def __init__(self, cbm, injection, generator, suffix, prefix=None):
        super().__init__()
        self.cbm = cbm
        self.injection = injection
        self.generator = generator
        self.suffix = suffix
        self.prefix = prefix

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen BatchNorm statistics must not change during generator training.
        self.suffix.eval()
        if self.prefix is not None:
            self.prefix.eval()
        return self

    def prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """For streamed internal noise, compute the frozen prefix per batch."""
        if self.prefix is None:
            return inputs
        with torch.no_grad():
            return self.prefix(inputs)

    def encode_concepts(self, inputs: torch.Tensor):
        return self.injection(self, inputs)

    def forward(self, inputs: torch.Tensor):
        concepts, energy = self.encode_concepts(inputs)
        return self.cbm.head(concepts), concepts, energy
