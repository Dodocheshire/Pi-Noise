"""Predict diagonal Gaussian parameters without access to class labels.

All generators consume B×L×D tokens. A global vector is one token;
spatial features and pooled images are sequences of tokens. Separate query
and text projections allow the backbone and CLIP text dimensions to differ.
"""
# reviewed
import math

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(channels, hidden), nn.GELU())

    # 注意，MLP没有用到text embedding
    def forward(self, tokens: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.network(tokens)


class ConceptCrossAttention(nn.Module):
    def __init__(self, channels: int, text_dim: int, hidden: int):
        super().__init__()
        self.query = nn.Linear(channels, hidden)
        self.key = nn.Linear(text_dim, hidden)
        self.value = nn.Linear(text_dim, hidden)

    def forward(self, tokens: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        query = self.query(tokens)
        scores = query @ self.key(text).T / math.sqrt(query.shape[-1])
        # The bank contains named concepts, never the sample's true class.
        return query + scores.softmax(dim=-1) @ self.value(text)


class NoiseGenerator(nn.Module):
    def __init__(
        self,
        channels: int,
        text: torch.Tensor,
        feature_scale: torch.Tensor,
        config: dict,
    ):
        super().__init__()
        self.parameterization = config["parameterization"]
        self.initial_scale = config["initial_scale"]
        self.image_grid = config["image_grid"]
        self.register_buffer("text", text)
        self.register_buffer("feature_scale", feature_scale)

        # A fixed Gaussian is deliberately parameter-free: no unused network.
        if self.parameterization == "fixed_gaussian":
            return
        hidden = config["hidden_dim"]
        if config["generator"] == "mlp":
            self.encoder = MLP(channels, hidden)
        else:
            self.encoder = ConceptCrossAttention(channels, text.shape[-1], hidden)
        self.mean_head = nn.Linear(hidden, channels)
        self.scale_head = nn.Linear(hidden, channels)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, math.log(math.expm1(self.initial_scale)))

    def distribution(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.parameterization == "fixed_gaussian":
            return torch.zeros_like(tokens), torch.ones_like(tokens) * (
                self.initial_scale * self.feature_scale
            )
        hidden = self.encoder(tokens / self.feature_scale, self.text)
        mean = self.mean_head(hidden) * self.feature_scale
        # This is a standard deviation, not a variance.
        scale = F.softplus(self.scale_head(hidden)) * self.feature_scale
        # 返回的是乘回各通道RMS的mean和scale
        return mean, scale

    def forward(self, tokens: torch.Tensor):
        mean, scale = self.distribution(tokens)
        perturbation = mean + scale * torch.randn_like(scale)
        # 噪声能量单位相对通道RMS倍数
        energy = (
            (mean / self.feature_scale).square() + (scale / self.feature_scale).square()
        ).mean()
        return tokens + perturbation, energy
