# model.py

import torch
from segmentation_models_pytorch import Unet

def get_model(device="cpu"):
    """
    Load pretrained U-Net model (ResNet34 encoder)
    """

    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # ⚠️ IMPORTANT (we load our trained weights)
        in_channels=1,
        classes=1
    )

    model = model.to(device)
    return model


def load_trained_model(model_path, device="cpu"):
    """
    Load saved model weights (.pth)
    """

    model = get_model(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model