
'''
This is just a placeholder code till I get the trained model added into the ml_model folder
'''

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

MODEL_PATH = "ml_models/deepguard_efficientnet_b0.pth"


def load_model():

    model = efficientnet_b0(weights=None)

    # Change classifier for 2 classes
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()

    return model