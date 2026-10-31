"""One training path shared by all target-side injection experiments."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .data import digest, prepare_data
from .evaluation import evaluate
from .generators import NoiseGenerator
from .injections import build_injection, channel_scale
from .inputs import ImagePaths
from .models import ConceptBottleneck, NoisyCBM
from .models.backbones import load_backbone
from .models.import_lfcbm import import_lfcbm
from .randomness import seed_all, seeded
from .training.adapt import adapt
from .training.lfcbm import fit_lfcbm


def cpu_state(module):
    # .detach() 返回一个不追踪梯度的新张量视图。如果原张量在gpu上，.cpu() 会真正拷贝一份到 CPU 内存，否则返回的 dict 和 module.state_dict() 指向同一块内存
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def source_hash() -> str:
    hasher = hashlib.sha256()
    for root in (Path("pi_cbm"), Path("scripts")):
        for path in sorted(root.rglob("*.py")):
            hasher.update(str(path).encode())
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def baseline_from_state(state, device):
    weight = state["projection.weight"]
    # 权重矩阵的列是in_feature，行是out_feature
    cbm = ConceptBottleneck(weight.shape[1], weight.shape[0], state["head.weight"].shape[0])
    cbm.load_state_dict(state)
    return cbm.to(device).eval().requires_grad_(False)


def get_baseline(data, cache_path, config, device):
    if config["baseline"]["import_dir"] is not None:
        return import_lfcbm(
            config["baseline"]["import_dir"], data, config["backbone"]["name"], device
        )
    baseline_seed = config["baseline"].get("seed")
    if baseline_seed is None:
        baseline_seed = config["experiment"]["seed"]
    identity = {
        "features": cache_path.name,
        "baseline": config["baseline"],
        "seed": baseline_seed,
        "implementation": digest(
            {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    Path("pi_cbm/training/lfcbm.py"),
                    *sorted(Path("pi_cbm/vendor").rglob("*.py")),
                ]
            }
        ),
    }
    path = Path(config["paths"]["cache"]) / f"baseline-{digest(identity)[:16]}.pt"
    if path.exists():
        saved = torch.load(path, weights_only=True)
        return baseline_from_state(saved["cbm"], device), saved["selected"], saved["metrics"]
    with seeded(baseline_seed):
        cbm, selected, metrics = fit_lfcbm(data, config, device)
    torch.save({"cbm": cpu_state(cbm), "selected": selected, "metrics": metrics}, path)
    return cbm, selected, metrics


def generator_channel_scale(data, key, config, backbone, device):
    """Calibrate noise in injected-feature units without retaining spatial inputs."""
    # 计算注入位置上训练集特征的逐通道 RMS（均方根），作为 NoiseGenerator.feature_scale
    if key == "features" or config["data"].get("input_mode", "cache") == "cache":
        return channel_scale(data["splits"]["train"][key])

    scale_path = Path(config["paths"]["cache"]) / (
        f"channel-scale-{digest([data['identity'], key])[:16]}.pt"
    )
    if scale_path.exists():
        return torch.load(scale_path, weights_only=True)

    # stream 模式是为了不落盘空间特征（ImageNet 存不下），所以 split[key] 不存在
    images = ImagePaths(data["splits"]["train"]["image_paths"], config["backbone"]["name"])
    loader = DataLoader(
        images,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=device.startswith("cuda"),
    )

    def feature_batches():
        with torch.no_grad():
            for image_batch in loader:
                image_batch = image_batch.to(device)
                # key == "internal" 时跑 backbone.prefix; key == "images" 时直接 yield 预处理后的图像本身
                yield backbone.prefix(image_batch) if key == "internal" else image_batch

    scale = channel_scale(feature_batches()) # 传入批次生成器
    torch.save(scale, scale_path)
    return scale


def build_model(cbm, data, selected, config, device):
    """
    @cbm: 已经拟合好并冻结的 ConceptBottleneck（get_baseline 的产物） 
    @selected: selected: 被选中的概念在概念库里的下标
    @data: prepare_data 的返回，提供 data["splits"][split][cache_key] 和 data["text"]
    """
    injection = build_injection(config["injection"]["site"])
    suffix = nn.Identity()
    prefix = None
    backbone = None
    if injection.cache_key in {"internal", "images"}:
        backbone = load_backbone(config["backbone"], device)
        suffix = backbone if injection.cache_key == "images" else backbone.suffix
        if injection.cache_key == "internal" and config["data"].get("input_mode") == "stream":
            # prefix 只在 stream 模式下才有值：stream 模式为了省磁盘不缓存中间特征，所以输入是原始图像，需要先用 backbone.prefix 现场算到注入点。cache 模式下输入就是缓存好的中间特征，prefix=None。
            prefix = backbone.prefix
    generator = None
    if config["injection"]["site"] != "baseline":
        scale = generator_channel_scale(data, injection.cache_key, config, backbone, device)
        # Initialization must not depend on whether feature/baseline caches
        # were freshly created or loaded earlier in this process.
        with seeded(config["experiment"]["seed"]):
            generator = NoiseGenerator(
                scale.numel(), # scale.numel() 是通道数：图像 3、内部特征 1024、全局特征 512
                data["text"][selected], # 只把被选中的 K 个概念的 CLIP embedding喂给生成器，作为交叉注意力的 key/value
                scale,
                config["noise"],
            ).to(device)
    return NoisyCBM(cbm, injection, generator, suffix, prefix).to(device)


def run_experiment(config: dict) -> Path:
    started = time.monotonic()
    seed_all(config["experiment"]["seed"])
    torch.set_num_threads(config["experiment"]["threads"])
    device = config["experiment"]["device"]
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + config["experiment"]["name"]
    run_dir = Path(config["paths"]["runs"]) / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    # Since development runs may precede the first Git commit, save actual source.
    for directory in ("pi_cbm", "scripts", "configs"):
        shutil.copytree(
            directory, run_dir / "source" / directory, ignore=shutil.ignore_patterns("__pycache__")
        )
    for filename in ("pyproject.toml", "requirements.txt"):
        shutil.copy2(filename, run_dir / "source" / filename)
    data, cache_path = prepare_data(config, device)
    cbm, selected, baseline_metrics = get_baseline(data, cache_path, config, device)
    clean_concepts = {}
    with torch.no_grad():
        for name, split in data["splits"].items():
            clean_concepts[name] = torch.cat(
                [cbm.concepts(batch.to(device)).cpu() for batch in split["features"].split(256)]
            )
    clean_model = NoisyCBM(cbm, build_injection("baseline"), None, nn.Identity())
    baseline_test = evaluate(
        clean_model, data["splits"]["test"], clean_concepts["test"], selected, config, device
    )
    print(f"Clean baseline test: {baseline_test}", flush=True)
    model = build_model(cbm, data, selected, config, device)
    seed_all(config["experiment"]["seed"])
    history, refit = adapt(model, data, clean_concepts, selected, config, device)
    with seeded(config["experiment"]["seed"] + 3000):
        metrics = evaluate(
            model, data["splits"]["test"], clean_concepts["test"], selected, config, device
        )
    torch.save(
        {
            "cbm": cpu_state(model.cbm),
            "generator": cpu_state(model.generator) if model.generator is not None else None,
            "selected": selected,
            "concepts": [data["concepts"][i] for i in selected],
            "baseline_concepts": {"test": clean_concepts["test"]},
            "cache_path": str(cache_path),
            "config": config,
        },
        run_dir / "checkpoint.pt",
    )
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest = {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_status": subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout,
        "source_sha256": source_hash(),
        "split_sha256": data["split_hash"],
        "concept_sha256": digest([data["concepts"][i] for i in selected]),
        "feature_cache": str(cache_path),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "gpu": torch.cuda.get_device_name(torch.device(device))
        if device.startswith("cuda")
        else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_seconds": time.monotonic() - started,
    }
    results = {
        "baseline_test": baseline_test,
        "test": metrics,
        "baseline_fit": baseline_metrics,
        "refit": refit,
        "history": history,
        "manifest": manifest,
    }
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (run_dir / "concepts.txt").write_text("\n".join(data["concepts"][i] for i in selected) + "\n")
    print(json.dumps({"run": str(run_dir), "test": metrics}, indent=2), flush=True)
    return run_dir
