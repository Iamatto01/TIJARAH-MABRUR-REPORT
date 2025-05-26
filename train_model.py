import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.cuda.amp import GradScaler, autocast
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ========== Dataset Class (Must be top-level for Windows multiprocessing) ==========
class QuranDataset(Dataset):
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
            label = self.data[self.data['filename'] == filename]['label'].values[0]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"\n⚠️ Error loading {img_path}: {str(e)}")
            dummy_img = Image.new('RGB', (self.img_size, self.img_size), color=(0,0,0))
            return self.transform(dummy_img), 0

def main():
    # === Configuration ===
    IMG_DIR = r"G:\Group Project\PY CODE\quranic-verse-detection\image"
    LABEL_CSV = "ayah_labels.csv"
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 3e-4
    IMG_SIZE = 224
    NUM_WORKERS = 0  # Windows requires 0 workers due to multiprocessing limitations
    MIXED_PRECISION = True

    # === GPU Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    scaler = GradScaler(enabled=MIXED_PRECISION)

    print("\n=== GPU Configuration ===")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    print(f"Mixed Precision Training: {MIXED_PRECISION}")

    # === Validate Paths ===
    print("\n=== Validating Paths ===")
    if not os.path.exists(IMG_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")
    if not os.path.exists(LABEL_CSV):
        raise FileNotFoundError(f"Label CSV not found: {LABEL_CSV}")
    print("✅ All paths validated")

    # === Load and preprocess labels ===
    print("\n=== Loading Labels ===")
    df = pd.read_csv(LABEL_CSV)
    print(f"Loaded {len(df)} entries from CSV")
    
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["surah_name"])
    print(f"Found {len(label_encoder.classes_)} unique classes")

    # === Train/val split ===
    print("\n=== Creating Train/Validation Split ===")
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df["label"], 
        random_state=42
    )
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # === Class Balancing ===
    print("\n=== Calculating Class Weights ===")
    class_counts = train_df["label"].value_counts().sort_index().tolist()
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    samples_weights = class_weights[train_df["label"].values]
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )

    # === Transforms ===
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

    # === DataLoaders ===
    print("\n=== Creating DataLoaders ===")
    train_ds = QuranDataset(train_df, IMG_DIR, train_transform, IMG_SIZE)
    val_ds = QuranDataset(val_df, IMG_DIR, val_transform, IMG_SIZE)
    
    print(f"Valid training images: {len(train_ds)}/{len(train_df)}")
    print(f"Valid validation images: {len(val_ds)}/{len(val_df)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # === Model Setup ===
    print("\n=== Initializing Model ===")
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, df["label"].nunique())
    model = model.to(device)
    print(f"Model initialized with {df['label'].nunique()} output classes")

    # === Training Setup ===
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.1, 
        patience=3
    )

    # === Training Loop ===
    print("\n=== Starting Training ===")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=MIXED_PRECISION):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            
            if batch_idx % 10 == 0:
                mem = torch.cuda.memory_allocated()/1024**2
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | GPU Mem: {mem:.1f}MB")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        # Epoch stats
        epoch_loss = running_loss / len(train_loader.dataset)
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        scheduler.step(val_acc)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
        print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")
        print(f"Val Accuracy: {val_acc*100:.2f}%")
        
        if device.type == 'cuda':
            mem_reserved = torch.cuda.memory_reserved()/1024**3
            mem_total = torch.cuda.get_device_properties(0).total_memory/1024**3
            print(f"GPU Memory Usage: {mem_reserved:.2f}/{mem_total:.2f} GB")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'acc': best_acc,
                'label_encoder': label_encoder,
                'classes': label_encoder.classes_
            }, "best_model.pth")
            print("💾 Saved new best model!")

    torch.save({
    'model_state_dict': model.state_dict(),
    'classes': label_encoder.classes_.tolist(),
    'input_size': IMG_SIZE,
    'model_arch': 'resnet50'
}, "best_model.pth")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()