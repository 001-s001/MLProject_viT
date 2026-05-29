import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss and accuracy curves from metrics.csv.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(metrics_path)

    plt.figure()
    plt.plot(data["epoch"], data["train_loss"], label="train_loss")
    plt.plot(data["epoch"], data["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(data["epoch"], data["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=200)
    plt.close()

    print(f"saved plots to {output_dir}")


if __name__ == "__main__":
    main()

