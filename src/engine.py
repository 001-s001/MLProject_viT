import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy_percent(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float(predictions.eq(targets).float().mean().item() * 100.0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1) * 100.0
    return avg_loss, avg_acc

