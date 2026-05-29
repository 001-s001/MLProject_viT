from typing import Any

from torch import nn

from src.models.simple_vit import SimpleViT


def build_model(config: dict[str, Any]) -> nn.Module:
    name = config.get("name", "simple_vit")
    kwargs = {key: value for key, value in config.items() if key != "name"}

    if name == "simple_vit":
        return SimpleViT(**kwargs)

    if name == "timm_vit":
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to use model.name=timm_vit") from exc
        model_name = kwargs.pop("model_name", "vit_tiny_patch16_224")
        pretrained = bool(kwargs.pop("pretrained", False))
        num_classes = int(kwargs.pop("num_classes"))
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)

    raise ValueError(f"unsupported model name: {name}")

