"""Reload a run and evaluate one sampled perturbation per image."""

import argparse
import json
from pathlib import Path

import torch

from pi_cbm.data import read_cache
from pi_cbm.evaluation import evaluate
from pi_cbm.evaluation.checkpoint import restore_model
from pi_cbm.randomness import seeded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    saved = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=True)
    config = saved["config"]
    device = args.device or config["experiment"]["device"]
    torch.set_num_threads(config["experiment"]["threads"])
    data = read_cache(saved["cache_path"])
    model = restore_model(saved, device)
    with seeded(config["experiment"]["seed"] + 3000):
        result = evaluate(
            model,
            data["splits"]["test"],
            saved["baseline_concepts"]["test"],
            saved["selected"],
            config,
            device,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
