import timm
import torch.nn as nn


def get_convnext_model():
    model = timm.create_model("convnext_base", pretrained=False, num_classes=2)
    return model
