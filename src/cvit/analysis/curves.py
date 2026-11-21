"""Plot training curves from one or more runs' ``metrics.json``.

Usage::

    python -m cvit.analysis.curves runs/vit-seed0 runs/cvit-seed0 --out results/curves.png

Also prints each run's final and peak test accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure


def load_metrics(run_dir: Path) -> list[dict[str, float]]:
    return json.loads((run_dir / "metrics.json").read_text())


def plot_curves(runs: dict[str, list[dict[str, float]]], out: Path) -> None:
    """Train loss, train accuracy and test accuracy per epoch, one line per run."""
    panels = [
        ("train_loss", "Train loss"),
        ("train_acc", "Train accuracy"),
        ("test_acc", "Test accuracy"),
    ]
    fig = Figure(figsize=(5 * len(panels), 4))
    axes = fig.subplots(1, len(panels))
    for ax, (key, title) in zip(axes, panels, strict=True):
        for name, history in runs.items():
            ax.plot([h["epoch"] for h in history], [h[key] for h in history], label=name)
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.3)
    axes[-1].legend()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot training curves of finished runs.")
    parser.add_argument("runs", nargs="+", type=Path, help="run directories with metrics.json")
    parser.add_argument("--out", type=Path, default=Path("results/curves.png"))
    args = parser.parse_args(argv)

    runs = {run.name: load_metrics(run) for run in args.runs}
    plot_curves(runs, args.out)
    for name, history in runs.items():
        best = max(history, key=lambda h: h["test_acc"])
        print(
            f"{name}: final test acc {history[-1]['test_acc']:.2%}, "
            f"peak {best['test_acc']:.2%} (epoch {best['epoch']})"
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
