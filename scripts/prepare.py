"""Download CIFAR and small pretrained weights; cache paired teacher/target features."""

import argparse

import torch

from pi_cbm.config import load_config
from pi_cbm.data import prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/cifar10_internal.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    torch.set_num_threads(config["experiment"]["threads"])
    data, path = prepare_data(config, config["experiment"]["device"])
    print(path)
    print({name: len(split["labels"]) for name, split in data["splits"].items()})


if __name__ == "__main__":
    main()
