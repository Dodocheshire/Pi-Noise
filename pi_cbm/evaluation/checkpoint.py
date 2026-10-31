"""Restore a trained adapter without loading its training feature cache."""

from pathlib import Path

import torch
from torch import nn

from ..generators import NoiseGenerator
from ..injections import build_injection
from ..models import ConceptBottleneck, NoisyCBM
from ..models.backbones import load_backbone
from .inference import predict


def restore_model(saved: dict, device: str, suffix: nn.Module | None = None) -> NoisyCBM:
    config, state = saved["config"], saved["cbm"]
    weight = state["projection.weight"]
    cbm = ConceptBottleneck(weight.shape[1], weight.shape[0], state["head.weight"].shape[0])
    cbm.load_state_dict(state)
    injection = build_injection(config["injection"]["site"])
    prefix = None
    if suffix is None:
        suffix = nn.Identity()
        if injection.cache_key in {"images", "internal"}:
            backbone = load_backbone(config["backbone"], device)
            suffix = backbone if injection.cache_key == "images" else backbone.suffix
            if injection.cache_key == "internal" and config["data"].get("input_mode") == "stream":
                prefix = backbone.prefix
    generator = None
    if saved["generator"] is not None:
        parameters = saved["generator"]
        generator = NoiseGenerator(
            parameters["feature_scale"].numel(),
            parameters["text"],
            parameters["feature_scale"],
            config["noise"],
        )
        generator.load_state_dict(parameters)
    return (
        NoisyCBM(cbm, injection, generator, suffix, prefix).to(device).eval().requires_grad_(False)
    )


class ImagePredictor(nn.Module):
    """Images must use the same target preprocessing as feature preparation.

    All interventions are expressed in standardized concept units. Setting a
    value to zero means the training mean, not necessarily semantic absence.
    """

    def __init__(self, run: str | Path, device: str):
        super().__init__()
        saved = torch.load(Path(run) / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.config = saved["config"]
        self.concepts = saved["concepts"]
        backbone = load_backbone(self.config["backbone"], device)
        site = self.config["injection"]["site"]
        if site == "target_image":
            self.encoder, suffix = nn.Identity(), backbone
        elif site == "target_internal":
            self.encoder, suffix = backbone.prefix, backbone.suffix
        else:
            self.encoder, suffix = backbone, nn.Identity()
        self.model = restore_model(saved, device, suffix)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images, intervention=None):
        inputs = self.encoder(images)
        return predict(self.model, inputs, intervention)
