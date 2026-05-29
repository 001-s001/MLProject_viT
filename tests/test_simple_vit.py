import pytest
import torch

from src.models.simple_vit import SimpleViT


def test_simple_vit_returns_logits_for_cifar_batch():
    model = SimpleViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )

    logits = model(torch.randn(3, 3, 32, 32))

    assert logits.shape == (3, 10)


def test_simple_vit_rejects_patch_size_that_does_not_tile_image():
    with pytest.raises(ValueError, match="image_size must be divisible"):
        SimpleViT(image_size=32, patch_size=5, num_classes=10)

