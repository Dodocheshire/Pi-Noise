"""Train one resolved configuration. Run from the repository root."""

import argparse

from pi_cbm.config import load_config
from pi_cbm.pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cifar10.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    run_experiment(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
