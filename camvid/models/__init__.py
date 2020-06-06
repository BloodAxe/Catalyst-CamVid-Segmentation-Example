from torch import nn

from . import fpn, linknet

__all__ = ["get_model", "MODEL_REGISTRY"]

MODEL_REGISTRY = {"resnet34_fpn": fpn.resnet34_fpn, "linknet34": linknet.LinkNet34}


def get_model(model_name: str, num_classes: int, dropout=0.0, pretrained=True) -> nn.Module:
    return MODEL_REGISTRY[model_name.lower()](num_classes=num_classes, dropout=dropout, pretrained=pretrained)
