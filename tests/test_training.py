import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine import evaluate_one_epoch, train_one_epoch
from src.models.simple_vit import SimpleViT


def test_train_and_evaluate_one_epoch_return_metrics():
    dataset = TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=2)
    model = SimpleViT(
        image_size=32,
        patch_size=8,
        num_classes=2,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate_one_epoch(model, loader, criterion, device)

    assert train_loss > 0
    assert val_loss > 0
    assert 0 <= val_acc <= 100

