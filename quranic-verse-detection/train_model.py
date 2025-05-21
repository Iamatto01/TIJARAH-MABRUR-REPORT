import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# === Configuration ===
IMG_DIR = "image"
LABEL_CSV = "ayah_labels.csv"
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 224
LEARNING_RATE = 1e-4

# === Step 1: Load and preprocess labels ===
df = pd.read_csv(LABEL_CSV)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["surah_name"])

# === Step 2: Train/val split ===
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# === Step 3: Custom Dataset ===
class QuranDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row["label"]
        return image, label

# === Step 4: Transforms ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize between -1 and 1
])

# === Step 5: DataLoaders ===
train_ds = QuranDataset(train_df, IMG_DIR, transform)
val_ds = QuranDataset(val_df, IMG_DIR, transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# === Step 6: CNN Model ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, df["label"].nunique())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Step 7: Training Setup ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Step 8: Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# === Step 9: Validation Accuracy ===
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

torch.save(model.state_dict(), "model.pth")
print(f"✅ Validation Accuracy: {100 * correct / total:.2f}%")
