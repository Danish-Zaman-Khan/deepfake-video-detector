'''
This is just a placeholder code till I get the trained model added into the ml_model folder
'''


import torch
from PIL import Image
from torchvision import transforms


# Same preprocessing used during model training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_frame(model, frame):
    """
    Predict whether a single video frame is Real or Fake.

    Args:
        model: Loaded EfficientNet-B0 model
        frame: OpenCV frame (BGR)

    Returns:
        label: Real or Fake
        confidence: prediction confidence
    """

    # OpenCV BGR → RGB
    frame = frame[:, :, ::-1]

    # NumPy array → PIL image
    image = Image.fromarray(frame)

    # Apply preprocessing
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Disable gradient calculation
    with torch.no_grad():

        output = model(image)

        # Convert output to probabilities
        probabilities = torch.softmax(output, dim=1)

        # Get highest probability
        confidence, predicted_class = torch.max(
            probabilities, dim=1
        )

    # Convert tensor to Python values
    predicted_class = predicted_class.item()
    confidence = confidence.item()

    # Class mapping
    if predicted_class == 0:
        label = "Real"
    else:
        label = "Fake"

    return label, confidence