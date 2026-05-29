import argparse
import shutil
from pathlib import Path

import torch
from torch import nn

from src.config import load_config
from src.datasets import build_cifar_loaders
from src.engine import evaluate_one_epoch, train_one_epoch
from src.models.factory import build_model
from src.utils.metrics import MetricsLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact ViT on CIFAR.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run one epoch for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = build_cifar_loaders(config["dataset"])
    model = build_model(config["model"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("training", {}).get("label_smoothing", 0.0)))

    training = config.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 3e-4)),
        weight_decay=float(training.get("weight_decay", 0.05)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training.get("epochs", 50)),
    )

    run_dir = Path(config.get("output_dir", "runs/vit_experiment"))
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
    (run_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    logger = MetricsLogger(run_dir / "metrics.csv")

    epochs = 1 if args.fast_dev_run else int(training.get("epochs", 50))
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        best_acc = max(best_acc, val_acc)
        logger.log(epoch, train_loss, val_loss, val_acc, best_acc, optimizer.param_groups[0]["lr"])

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_acc": best_acc,
            "class_names": class_names,
            "config": config,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_acc >= best_acc:
            torch.save(checkpoint, run_dir / "best.pt")

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f} best_acc={best_acc:.2f}"
        )


if __name__ == "__main__":
    main()

