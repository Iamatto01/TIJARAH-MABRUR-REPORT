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
from torchvision.models import resnet50, ResNet50_Weights
from torch.cuda.amp import GradScaler, autocast
import warnings

warnings.filterwarnings("ignore")

class MultiLabelQuranDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, img_size=224):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
        self.valid_files = [f for f in dataframe['filename'] if os.path.exists(os.path.join(img_dir, f))]

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        filename = self.valid_files[idx]
        img_path = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            row = self.data[self.data['filename'] == filename].iloc[0]
            label_surah = row['label_surah']
            label_classification = row['label_classification']
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor([label_surah, label_classification], dtype=torch.long)
        except Exception as e:
            print(f"\n⚠️ Error loading {img_path}: {str(e)}")
            dummy_img = Image.new('RGB', (self.img_size, self.img_size), color=(0, 0, 0))
            return self.transform(dummy_img), torch.tensor([0, 0], dtype=torch.long)

def main():
    IMG_DIR = r"G:\\Group Project\\PY CODE\\quranic-verse-detection\\image"
    LABEL_CSV = r"G:\Group Project\PY CODE\quranic-verse-detection\ayah_labels.csv"
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 3e-4
    IMG_SIZE = 224
    NUM_WORKERS = 0
    MIXED_PRECISION = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(enabled=MIXED_PRECISION)

    df = pd.read_csv(LABEL_CSV)

    surah_encoder = LabelEncoder()
    class_encoder = LabelEncoder()
    df["label_surah"] = surah_encoder.fit_transform(df["surah_name"])
    df["label_classification"] = class_encoder.fit_transform(df["classification"])

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label_surah"], random_state=42)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.RandomRotation(5),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = MultiLabelQuranDataset(train_df, IMG_DIR, train_transform, IMG_SIZE)
    val_ds = MultiLabelQuranDataset(val_df, IMG_DIR, val_transform, IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    features = base_model.fc.in_features
    base_model.fc = nn.Identity()
    base_model = base_model.to(device)

    classifier_surah = nn.Linear(features, len(surah_encoder.classes_)).to(device)
    classifier_classification = nn.Linear(features, len(class_encoder.classes_)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(list(base_model.parameters()) + 
                                  list(classifier_surah.parameters()) + 
                                  list(classifier_classification.parameters()), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        base_model.train()
        classifier_surah.train()
        classifier_classification.train()

        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with autocast(enabled=MIXED_PRECISION):
                features_out = base_model(images)
                out_surah = classifier_surah(features_out)
                out_classification = classifier_classification(features_out)
                loss1 = criterion(out_surah, labels[:, 0])
                loss2 = criterion(out_classification, labels[:, 1])
                loss = loss1 + loss2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
