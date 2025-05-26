import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image
import joblib

def load_model(model_path):
    """Load the trained model and label encoder"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Recreate model architecture
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint['label_encoder_classes']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Recreate label encoder
    label_encoder = joblib.load(checkpoint['label_encoder'])
    
    return model, label_encoder

def predict_verse(model, label_encoder, image_tensor, device):
    """Make prediction on processed image tensor"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
        
        predicted_idx = preds.item()
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
        confidence_percent = confidence.item() * 100
        
    return predicted_label, confidence_percent