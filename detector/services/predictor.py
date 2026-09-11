import torch
from PIL import Image
from torchvision import transforms
from .model_loader import get_model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def predict_face(image_path):
    model = get_model()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        # Apply Sigmoid to single logit output
        fake_probability = torch.sigmoid(output).item()

    return {
        'is_fake': fake_probability > 0.5,
        'confidence': round(fake_probability, 4)
    }