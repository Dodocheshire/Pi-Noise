"""Small YAML composition: later files/overrides win, resolved config is saved per run."""

from copy import deepcopy
from pathlib import Path

import yaml


def merge(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, overrides: list[str] = ()) -> dict:
    path = Path(path)
    document = yaml.safe_load(path.read_text())
    result = {}
    for parent in document.pop("extends", []):
        result = merge(result, load_config(path.parent / parent))
    result = merge(result, document)
    for override in overrides:
        dotted_key, value = override.split("=", 1)
        cursor = result
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            cursor = cursor[key]
        if keys[-1] not in cursor:
            raise KeyError(f"Unknown configuration key: {dotted_key}")
        cursor[keys[-1]] = yaml.safe_load(value)
    validate(result)
    return result


def validate(config: dict) -> None:
    """Catch incompatible experiments at the CLI boundary, before downloading data."""
    # Partial files are valid while inheritance is being resolved.
    if "experiment" not in config or "training" not in config:
        return
    choices = {
        ("data", "dataset"): {"cifar10", "cifar100", "cub", "places365", "imagenet"},
        ("injection", "site"): {"baseline", "target_image", "target_global", "target_internal"},
        ("noise", "parameterization"): {"fixed_gaussian", "mean_scale"},
        ("noise", "generator"): {"mlp", "concept_cross_attention"},
        ("training", "stage"): {"generator_only", "refit_head", "finetune_projection"},
    }
    for (section, key), values in choices.items():
        if config[section][key] not in values:
            raise ValueError(f"{section}.{key} must be one of {sorted(values)}")
    input_mode = config["data"].get("input_mode", "cache")
    if input_mode not in {"cache", "stream"}:
        raise ValueError("data.input_mode must be cache or stream")
    if input_mode == "stream" and config["data"]["dataset"] not in {"cub", "imagenet"}:
        raise ValueError("Streamed inputs require ImageFolder source images")
    if config["training"]["epochs"] < 1 or config["noise"]["initial_scale"] <= 0:
        raise ValueError("training.epochs and noise.initial_scale must be positive")
    if config["injection"]["site"] == "target_image" and not (
        config["data"]["cache_images"] or input_mode == "stream"
    ):
        raise ValueError(
            "Image injection requires data.cache_images=true or data.input_mode=stream"
        )
    if config["injection"]["site"] == "target_image" and config["injection"]["point"] != "input":
        raise ValueError("Image injection requires point=input")
    if config["injection"]["site"] == "target_internal":
        expected = {
            "resnet18": "pre_layer4",
            "resnet50": "pre_layer4",
            "resnet18_cub": "pre_stage4",
            "clip_RN50": "pre_attnpool",
        }[config["backbone"]["name"]]
        if config["injection"]["point"] != expected or not (
            config["data"]["cache_internal"] or input_mode == "stream"
        ):
            raise ValueError(
                f"Internal injection requires point={expected} and cached or streamed inputs"
            )
    if (
        config["injection"]["site"] == "baseline"
        and config["training"]["stage"] == "finetune_projection"
    ):
        raise ValueError("baseline supports a frozen CBM or a clean head refit")
