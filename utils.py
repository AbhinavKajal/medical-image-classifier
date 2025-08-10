import os
import shutil
import torch
import torch.nn as nn
from torchvision.models import resnet50

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def build_resnet50(num_classes=2):
    model = resnet50(weights=None)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    return model

def load_checkpoint(path: str, device: torch.device = torch.device("cpu")):
    """
    Returns a ResNet-50 model with checkpoint loaded.
    Accepts either raw state_dict or dict with 'state_dict'/'model_state_dict'.
    """
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
    else:
        state_dict = state

    model = build_resnet50(num_classes=2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
