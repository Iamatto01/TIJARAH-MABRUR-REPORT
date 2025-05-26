import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

# Config
MODEL_PATH = r"G:\Group Project\PY CODE\quranic-verse-detection\models\best_model.pth"  # Your trained model
IMG_PATH = r"G:\Group Project\PY CODE\quranic-verse-detection\data\raw_images\Test.png"
IMG_SIZE = 224

# 1. Load YOUR model (no pretrained weights)
model = resnet50(weights=None)  # No download!

# 2. Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model.fc = nn.Linear(model.fc.in_features, len(checkpoint['label_encoder_classes']))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Prediction
image = Image.open(IMG_PATH).convert("RGB")
image = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image)
    _, pred = torch.max(output, 1)
    print(f"Predicted class: {pred.item()}")