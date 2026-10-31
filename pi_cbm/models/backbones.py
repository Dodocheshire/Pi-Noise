"""Explicit backbone splits keep a differentiable nonlinear suffix after noise."""
# reviewed
import clip
import torch
from torch import nn
from torchvision import models, transforms


class ResNetSplit(nn.Module):
    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        # Match LF-CBM's unsuffixed ResNet names (V1 weights and preprocessing).
        network = models.get_model(name, weights="IMAGENET1K_V1" if pretrained else None)
        self.prefix = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
        )
        self.suffix = nn.Sequential(network.layer4, network.avgpool, nn.Flatten(1))
        self.feature_dim = network.fc.in_features
        self.internal_channels = 256 if name == "resnet18" else 1024
        self.point = "pre_layer4"

    def forward(self, images):
        return self.suffix(self.prefix(images))


class Cast(nn.Module):
    """Keep native CLIP precision inside the frozen encoder and FP32 at the CBM."""

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, value):
        return value.to(self.dtype)


class CLIPResNetSplit(nn.Module):
    def __init__(self, visual: nn.Module):
        super().__init__()
        # CLIP's anti-aliased stem differs from torchvision's ResNet stem.
        self.prefix = nn.Sequential(
            Cast(visual.conv1.weight.dtype),
            visual.conv1,
            visual.bn1,
            visual.relu1,
            visual.conv2,
            visual.bn2,
            visual.relu2,
            visual.conv3,
            visual.bn3,
            visual.relu3,
            visual.avgpool,
            visual.layer1,
            visual.layer2,
            visual.layer3,
            visual.layer4,
        )
        self.suffix = nn.Sequential(
            Cast(visual.conv1.weight.dtype), visual.attnpool, Cast(torch.float32)
        )
        self.feature_dim = visual.output_dim
        self.internal_channels = visual.attnpool.k_proj.in_features
        self.point = "pre_attnpool"

    def forward(self, images):
        return self.suffix(self.prefix(images))


class CUBResNetSplit(nn.Module):
    """Use LF-CBM's CUB-trained pytorchcv ResNet18, split before stage4."""

    def __init__(self, network: nn.Module):
        super().__init__()
        features = network.features
        self.prefix = nn.Sequential(
            features.init_block, features.stage1, features.stage2, features.stage3
        )
        self.suffix = nn.Sequential(features.stage4, features.final_pool, nn.Flatten(1))
        self.feature_dim = network.output.in_features
        self.internal_channels = 256
        self.point = "pre_stage4"

    def forward(self, images):
        return self.suffix(self.prefix(images))


def target_preprocess(name: str):
    """Match each pretrained backbone's original image preprocessing."""
    if name == "clip_RN50":
        return clip.clip._transform(224)
    if name == "resnet18_cub":
        # This is get_resnet_imagenet_preprocess from the official LF-CBM.
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return models.get_model_weights(name).IMAGENET1K_V1.transforms()


def load_backbone(config: dict, device: str) -> nn.Module:
    if config["name"] == "clip_RN50":
        native = config.get("precision", "float32") == "native"
        network, _ = clip.load("RN50", device=device if native else "cpu", jit=False)
        backbone = CLIPResNetSplit(network.visual if native else network.visual.float())
    elif config["name"] == "resnet18_cub":
        from pytorchcv.model_provider import get_model

        backbone = CUBResNetSplit(get_model("resnet18_cub", pretrained=True))
    else:
        if config["name"] not in {"resnet18", "resnet50"}:
            raise ValueError("Supported backbones: resnet18, resnet50, resnet18_cub, clip_RN50")
        backbone = ResNetSplit(config["name"])
    return backbone.to(device).eval().requires_grad_(False)
