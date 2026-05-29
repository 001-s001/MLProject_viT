import argparse
from pathlib import Path

import torch
from torch import nn

from src.datasets import build_cifar_loaders
from src.engine import evaluate_one_epoch
from src.models.factory import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ViT checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _ = build_cifar_loaders(config["dataset"])
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion, device)
    print(f"checkpoint={checkpoint_path} val_loss={val_loss:.4f} val_acc={val_acc:.2f}")


if __name__ == "__main__":
    main()

