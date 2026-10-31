"""Read the original LF-CBM's five tensors without importing its training code."""
# reviewed
import json
from pathlib import Path

import torch

from .cbm import ConceptBottleneck


def import_lfcbm(directory, data, backbone_name, device):
    directory = Path(directory)
    arguments = json.loads((directory / "args.txt").read_text())
    if arguments["backbone"] != backbone_name:
        raise ValueError("Imported CBM and target feature backbone must match")
    names = (directory / "concepts.txt").read_text().splitlines()
    lookup = {name: index for index, name in enumerate(data["concepts"])}
    # lfcbm官方过滤后的concept名，转换为数据集concept文件（例如：concepts: configs/concepts/cifar10_official.txt）的concept id
    selected = torch.tensor([lookup[name] for name in names])
    tensors = {
        name: torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)
        for name in ("W_c", "W_g", "b_g", "proj_mean", "proj_std")
    }
    cbm = ConceptBottleneck(
        data["splits"]["train"]["features"].shape[1], len(names), len(data["classes"])
    )
    cbm.load_state_dict(
        {
            "projection.weight": tensors["W_c"],
            "head.weight": tensors["W_g"],
            "head.bias": tensors["b_g"],
            "concept_mean": tensors["proj_mean"].flatten(),
            "concept_std": tensors["proj_std"].flatten(),
        }
    )
    # Importing a model does not establish how its original train/val split was
    # chosen. The caller must report that provenance separately.
    return cbm.to(device).eval().requires_grad_(False), selected, {"imported_from": str(directory)}
