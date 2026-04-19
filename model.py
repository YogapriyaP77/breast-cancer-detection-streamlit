import torch
import torch.nn as nn
from torchvision import models

# Create model architecture
def get_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# Load trained weights
def load_model():
    model = get_model()

    model.load_state_dict(
        torch.load("model.pth", map_location=torch.device('cpu'))
    )

    model.eval()
    return model