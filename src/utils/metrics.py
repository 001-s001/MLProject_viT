import csv
from pathlib import Path


class MetricsLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = ["epoch", "train_loss", "val_loss", "val_acc", "best_acc", "lr"]
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        best_acc: float,
        lr: float,
    ) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_acc": best_acc,
                    "lr": lr,
                }
            )

