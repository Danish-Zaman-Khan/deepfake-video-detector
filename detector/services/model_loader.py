import os
import torch
import torchvision.models as models
from django.conf import settings

_model = None

def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(settings.BASE_DIR, 'ml_model', 'resnet50_best.pth')
        
        # Instantiate ResNet50 base
        _model = models.resnet50(weights=None)
        
        # Match checkpoint shape: 1 output neuron instead of 2
        num_ftrs = _model.fc.in_features
        _model.fc = torch.nn.Linear(num_ftrs, 1)
        
        # Load checkpoint
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        _model.load_state_dict(state_dict)
        _model.eval()
        
    return _model