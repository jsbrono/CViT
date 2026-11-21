"""Print a Markdown table of parameter counts for model configs.

Usage::

    python -m cvit.analysis.params configs/vit.yaml configs/cvit.yaml

No training or data is needed; the models are only instantiated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cvit.model import ViT, count_parameters
from cvit.train import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parameter counts and sizes of model configs.")
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args(argv)

    print(f"| {'Config':<28} | {'Parameters':>12} | {'Real-equivalent':>15} | {'Size (MB)':>9} |")
    print(f"|{'-' * 30}|{'-' * 13}:|{'-' * 16}:|{'-' * 10}:|")
    for path in args.configs:
        model_config, _ = load_config(path)
        stats = count_parameters(ViT(model_config))
        print(
            f"| {path.stem:<28} | {stats['total']:>12,} | {stats['real_equivalent']:>15,} "
            f"| {stats['megabytes']:>9.2f} |"
        )


if __name__ == "__main__":
    main()
