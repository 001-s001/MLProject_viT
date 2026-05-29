from pathlib import Path

from src.config import load_config
from src.models.factory import build_model


def test_load_config_reads_yaml_file(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  name: cifar10
model:
  name: simple_vit
  num_classes: 10
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["dataset"]["name"] == "cifar10"
    assert config["model"]["num_classes"] == 10


def test_build_model_creates_simple_vit_from_config():
    model = build_model(
        {
            "name": "simple_vit",
            "image_size": 32,
            "patch_size": 8,
            "num_classes": 100,
            "embed_dim": 96,
            "depth": 2,
            "num_heads": 3,
        }
    )

    assert model.head.out_features == 100
    assert model.patch_embed.patch_size == 8

