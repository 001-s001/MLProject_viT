import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets import build_cifar_loaders
from src.models.factory import build_model

@dataclass(frozen=True)
class FailureCase:
    index: int
    target: int
    predicted: int
    confidence: float


def select_top_failures(logits: torch.Tensor, targets: torch.Tensor, top_k: int = 30) -> list[FailureCase]:
    probabilities = torch.softmax(logits.detach().cpu(), dim=1)
    confidences, predictions = probabilities.max(dim=1)
    targets = targets.detach().cpu()
    mistakes = predictions.ne(targets)

    cases = [
        FailureCase(
            index=int(i),
            target=int(targets[i]),
            predicted=int(predictions[i]),
            confidence=float(confidences[i]),
        )
        for i in torch.nonzero(mistakes, as_tuple=False).flatten()
    ]
    cases.sort(key=lambda case: case.confidence, reverse=True)
    return cases[:top_k]


def save_failure_report(cases: list[FailureCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["index,target,predicted,confidence"]
    lines.extend(f"{c.index},{c.target},{c.predicted},{c.confidence:.6f}" for c in cases)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_targets = []
    for images, targets in loader:
        logits = model(images.to(device)).cpu()
        all_logits.append(logits)
        all_targets.append(targets.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export high-confidence ViT failure cases.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output", default=None)
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

    logits, targets = collect_logits(model, val_loader, device)
    cases = select_top_failures(logits, targets, args.top_k)
    output = Path(args.output) if args.output else checkpoint_path.parent / "failure_cases.csv"
    save_failure_report(cases, output)
    print(f"saved {len(cases)} failure cases to {output}")


if __name__ == "__main__":
    main()
