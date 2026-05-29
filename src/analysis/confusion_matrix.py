import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from src.analysis.failure_cases import collect_logits
from src.datasets import build_cifar_loaders
from src.models.factory import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a confusion matrix for a ViT checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    class_names = checkpoint["class_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _ = build_cifar_loaders(config["dataset"])
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    logits, targets = collect_logits(model, val_loader, device)
    preds = logits.argmax(dim=1).numpy()
    cm = confusion_matrix(targets.numpy(), preds, labels=list(range(len(class_names))))

    output = Path(args.output) if args.output else checkpoint_path.parent / "confusion_matrix.png"
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("ViT Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"saved confusion matrix to {output}")


if __name__ == "__main__":
    main()

