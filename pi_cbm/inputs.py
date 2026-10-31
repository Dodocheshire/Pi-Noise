"""Serve cached features or decode source images for one model input batch."""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader

from .models.backbones import target_preprocess


class ImagePaths(Dataset):
    """Decode only the images requested by a training or evaluation batch."""

    def __init__(self, paths: list[str], backbone_name: str):
        self.paths = paths
        self.transform = target_preprocess(backbone_name)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(default_loader(self.paths[index]))


class ModelInputs(Dataset):
    """Keep cached labels and concept targets aligned with their source image."""

    def __init__(self, split: dict, model, config: dict, *fields: torch.Tensor):
        key = model.injection.cache_key
        self.stream = config["data"].get("input_mode", "cache") == "stream" and key != "features"
        self.inputs = (
            ImagePaths(split["image_paths"], config["backbone"]["name"])
            if self.stream
            else split[key]
        )
        self.fields = fields

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return (self.inputs[index], *(field[index] for field in self.fields))


def input_loader(
    split: dict,
    model,
    config: dict,
    *fields: torch.Tensor,
    batch_size: int,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = ModelInputs(split, model, config, *fields)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=config["data"].get("num_workers", 0) if dataset.stream else 0,
        pin_memory=config["experiment"]["device"].startswith("cuda"),
    )
