"""
CropGardian — ResNet-50 Training Script
========================================
Fine-tunes ResNet-50 on PlantVillage dataset (38 disease classes).

USAGE:
    # 1. Generate synthetic dataset (if no real PlantVillage data):
    cd backend/data && python generate_dataset.py --samples 200
    # (or use --full for 500/class, or provide real PlantVillage data in data/plantvillage/)

    # 2. Train:
    cd backend
    python model/train.py

    # 3. Predict single image:
    python model/train.py predict path/to/leaf.jpg

REAL PlantVillage dataset:
    kaggle datasets download -d emmarex/plantdisease
    # Extract to: backend/data/plantvillage/
    # It must have subfolders named exactly as in class_names.json
"""

import os
import json
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent          # backend/
DATA_DIR   = BASE_DIR / "data" / "plantvillage"    # backend/data/plantvillage/
MODEL_DIR  = Path(__file__).parent                 # backend/model/
MODEL_PATH = MODEL_DIR / "cropgardian_resnet50.pth"
CLASS_JSON = MODEL_DIR / "class_names.json"
HIST_JSON  = MODEL_DIR / "training_history.json"

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "num_epochs"     : 30,
    "batch_size"     : 32,
    "learning_rate"  : 1e-3,
    "weight_decay"   : 1e-4,
    "val_split"      : 0.20,
    "num_workers"    : 4,
    "image_size"     : 224,
    "freeze_epochs"  : 10,      # epochs to train only FC head (frozen backbone)
    "label_smoothing": 0.1,
    "dropout_fc"     : 0.5,
    "dropout_mid"    : 0.3,
    "patience"       : 7,       # early stopping patience
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── PlantVillage 38-class ground-truth label map ───────────────────────────────
# Maps folder name → (crop, disease) — for display/API use
LABEL_MAP = {
    "Apple___Apple_scab"                                  : ("Apple",      "Apple Scab"),
    "Apple___Black_rot"                                   : ("Apple",      "Black Rot"),
    "Apple___Cedar_apple_rust"                            : ("Apple",      "Cedar Apple Rust"),
    "Apple___healthy"                                     : ("Apple",      "Healthy"),
    "Blueberry___healthy"                                 : ("Blueberry",  "Healthy"),
    "Cherry_(including_sour)___Powdery_mildew"            : ("Cherry",     "Powdery Mildew"),
    "Cherry_(including_sour)___healthy"                   : ("Cherry",     "Healthy"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"  : ("Corn",       "Gray Leaf Spot"),
    "Corn_(maize)___Common_rust_"                         : ("Corn",       "Common Rust"),
    "Corn_(maize)___Northern_Leaf_Blight"                 : ("Corn",       "Northern Leaf Blight"),
    "Corn_(maize)___healthy"                              : ("Corn",       "Healthy"),
    "Grape___Black_rot"                                   : ("Grape",      "Black Rot"),
    "Grape___Esca_(Black_Measles)"                        : ("Grape",      "Black Measles (Esca)"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"          : ("Grape",      "Leaf Blight"),
    "Grape___healthy"                                     : ("Grape",      "Healthy"),
    "Orange___Haunglongbing_(Citrus_greening)"            : ("Orange",     "Citrus Greening (HLB)"),
    "Peach___Bacterial_spot"                              : ("Peach",      "Bacterial Spot"),
    "Peach___healthy"                                     : ("Peach",      "Healthy"),
    "Pepper,_bell___Bacterial_spot"                       : ("Pepper",     "Bacterial Spot"),
    "Pepper,_bell___healthy"                              : ("Pepper",     "Healthy"),
    "Potato___Early_blight"                               : ("Potato",     "Early Blight"),
    "Potato___Late_blight"                                : ("Potato",     "Late Blight"),
    "Potato___healthy"                                    : ("Potato",     "Healthy"),
    "Raspberry___healthy"                                 : ("Raspberry",  "Healthy"),
    "Rice___Blast"                                        : ("Rice",       "Blast Disease"),
    "Rice___healthy"                                      : ("Rice",       "Healthy"),
    "Soybean___healthy"                                   : ("Soybean",    "Healthy"),
    "Squash___Powdery_mildew"                             : ("Squash",     "Powdery Mildew"),
    "Strawberry___Leaf_scorch"                            : ("Strawberry", "Leaf Scorch"),
    "Strawberry___healthy"                                : ("Strawberry", "Healthy"),
    "Tomato___Bacterial_spot"                             : ("Tomato",     "Bacterial Spot"),
    "Tomato___Early_blight"                               : ("Tomato",     "Early Blight"),
    "Tomato___Late_blight"                                : ("Tomato",     "Late Blight"),
    "Tomato___Leaf_Mold"                                  : ("Tomato",     "Leaf Mold"),
    "Tomato___Septoria_leaf_spot"                         : ("Tomato",     "Septoria Leaf Spot"),
    "Tomato___Spider_mites Two-spotted_spider_mite"       : ("Tomato",     "Spider Mites"),
    "Tomato___Target_Spot"                                : ("Tomato",     "Target Spot"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"              : ("Tomato",     "Yellow Leaf Curl Virus"),
    "Tomato___Tomato_mosaic_virus"                        : ("Tomato",     "Mosaic Virus"),
    "Tomato___healthy"                                    : ("Tomato",     "Healthy"),
}

# ── Transforms ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(CONFIG["image_size"], scale=(0.65, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.15),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.08),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.10, scale=(0.02, 0.1)),   # simulate occlusion
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Dataset ─────────────────────────────────────────────────────────────────────
def load_data():
    if not DATA_DIR.exists():
        print(f"\n❌  Dataset not found at: {DATA_DIR}")
        print("Run: cd backend/data && python generate_dataset.py --samples 200")
        print("Or download real PlantVillage and extract to backend/data/plantvillage/\n")
        sys.exit(1)

    print(f"\n📂  Loading dataset from: {DATA_DIR}")
    full_ds = datasets.ImageFolder(str(DATA_DIR), transform=train_transform)
    class_names = full_ds.classes
    num_classes = len(class_names)
    print(f"    Classes found : {num_classes}")
    print(f"    Total images  : {len(full_ds):,}")
    print(f"    First 5       : {class_names[:5]}")

    # Save class names
    with open(CLASS_JSON, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"    class_names.json saved → {CLASS_JSON}")

    # Train / val split
    n_val   = int(len(full_ds) * CONFIG["val_split"])
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Apply val transform separately
    val_ds.dataset = datasets.ImageFolder(str(DATA_DIR), transform=val_transform)

    # Weighted sampler for class balance
    targets      = [full_ds.targets[i] for i in train_ds.indices]
    class_counts = np.bincount(targets, minlength=num_classes)
    class_weights= 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              sampler=sampler, num_workers=CONFIG["num_workers"],
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"],
                              pin_memory=True)

    print(f"    Train: {n_train:,}  |  Val: {n_val:,}")
    return train_loader, val_loader, class_names


# ── Model ───────────────────────────────────────────────────────────────────────
def build_model(num_classes):
    print(f"\n🏗️   Building ResNet-50 (pretrained ImageNet) → {num_classes} classes")
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            print("    ⚠️  Cannot download pretrained weights (network restricted). Using random init.")
            print("       For full accuracy, run on a machine with internet access.")
            model = models.resnet50(weights=None)

    # Freeze backbone initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace FC head
    num_ftrs = model.fc.in_features   # 2048
    model.fc = nn.Sequential(
        nn.Dropout(CONFIG["dropout_fc"]),
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(CONFIG["dropout_mid"]),
        nn.Linear(512, num_classes),
    )
    return model.to(DEVICE)


# ── Training helpers ────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type=="cuda"):
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total   += labels.size(0)
        if batch_idx % 50 == 0:
            print(f"    batch {batch_idx:4d}/{len(loader)} | loss {loss.item():.4f}", end="\r")
    return running_loss / total, 100.0 * correct / total


def val_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, 100.0 * correct / total


# ── Main training loop ──────────────────────────────────────────────────────────
def train():
    print("\n" + "="*65)
    print("  CropGardian — ResNet-50 Training on PlantVillage Dataset")
    print("="*65)
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {CONFIG['num_epochs']}")
    print(f"  Batch size : {CONFIG['batch_size']}")
    print(f"  Freeze for : {CONFIG['freeze_epochs']} epochs (warm-up)")

    train_loader, val_loader, class_names = load_data()
    num_classes = len(class_names)
    model       = build_model(num_classes)

    criterion   = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    optimizer   = optim.AdamW(model.fc.parameters(),
                              lr=CONFIG["learning_rate"],
                              weight_decay=CONFIG["weight_decay"])
    scheduler   = CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6)
    scaler      = torch.amp.GradScaler("cuda", enabled=DEVICE.type=="cuda")

    best_val_acc = 0.0
    patience_cnt = 0
    history      = []

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(CONFIG["num_epochs"]):
        t0 = time.time()

        # ── Unfreeze backbone after warm-up ────────────────────────────────
        if epoch == CONFIG["freeze_epochs"]:
            print(f"\n🔓  Epoch {epoch+1}: Unfreezing entire backbone for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(),
                                    lr=CONFIG["learning_rate"] * 0.1,
                                    weight_decay=CONFIG["weight_decay"])
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=CONFIG["num_epochs"] - epoch,
                                          eta_min=1e-7)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss,   val_acc   = val_epoch(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"\nEpoch [{epoch+1:2d}/{CONFIG['num_epochs']}]  ({elapsed:.0f}s)")
        print(f"  Train  →  Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%")
        print(f"  Val    →  Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%", end="")

        history.append({"epoch": epoch+1,
                        "train_loss": round(train_loss,4), "train_acc": round(train_acc,2),
                        "val_loss":   round(val_loss,4),   "val_acc":   round(val_acc,2)})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ Best model saved ({val_acc:.2f}%)")
        else:
            patience_cnt += 1
            print(f"  (patience {patience_cnt}/{CONFIG['patience']})")
            if patience_cnt >= CONFIG["patience"]:
                print(f"\n⏹️  Early stopping — no improvement for {CONFIG['patience']} epochs")
                break

    # Save history
    with open(HIST_JSON, "w") as f:
        json.dump({"config": CONFIG, "device": str(DEVICE),
                   "num_classes": num_classes, "class_names": class_names,
                   "best_val_acc": best_val_acc, "history": history}, f, indent=2)

    print(f"\n{'='*65}")
    print(f"✅  Training complete!")
    print(f"    Best Val Accuracy : {best_val_acc:.2f}%")
    print(f"    Model saved to    : {MODEL_PATH}")
    print(f"    History saved to  : {HIST_JSON}")
    print(f"{'='*65}\n")
    return model


# ── Single image prediction ─────────────────────────────────────────────────────
def predict(image_path, model_path=None):
    """Run inference on a single image and print result."""
    from PIL import Image as PILImage

    mp = Path(model_path) if model_path else MODEL_PATH
    if not mp.exists():
        print(f"❌  Model not found: {mp}\nTrain first: python model/train.py")
        return

    if not CLASS_JSON.exists():
        print(f"❌  class_names.json not found: {CLASS_JSON}")
        return

    with open(CLASS_JSON) as f:
        class_names = json.load(f)

    model = build_model(len(class_names))
    model.load_state_dict(torch.load(mp, map_location=DEVICE))
    model.eval()

    transform = val_transform
    img = PILImage.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs[0], dim=0)
        conf, pred = torch.max(probs, 0)

    folder_name  = class_names[pred.item()]
    confidence   = conf.item() * 100
    crop, disease = LABEL_MAP.get(folder_name, (folder_name, folder_name))

    print(f"\n{'='*50}")
    print(f"  Crop    : {crop}")
    print(f"  Disease : {disease}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"  Class   : {folder_name}")
    print(f"{'='*50}")

    # Top-5
    top5_probs, top5_idx = torch.topk(probs, min(5, len(class_names)))
    print("\nTop-5 predictions:")
    for i, (p, idx) in enumerate(zip(top5_probs, top5_idx)):
        cn = class_names[idx.item()]
        cr, di = LABEL_MAP.get(cn, (cn, cn))
        print(f"  {i+1}. {di} ({cr}) — {p.item()*100:.1f}%")

    return {"crop": crop, "disease": disease, "confidence": confidence, "class": folder_name}


# ── Evaluate on validation set ──────────────────────────────────────────────────
def evaluate():
    """Load saved model and run full validation with per-class accuracy."""
    _, val_loader, class_names = load_data()
    model = build_model(len(class_names))
    if not MODEL_PATH.exists():
        print(f"❌  Model not found. Train first: python model/train.py")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = val_epoch(model, val_loader, criterion)
    print(f"\nValidation Loss: {val_loss:.4f}  |  Accuracy: {val_acc:.2f}%")


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        cmd = sys.argv[1].lower()
        if cmd == "predict" and len(sys.argv) >= 3:
            predict(sys.argv[2], model_path=sys.argv[3] if len(sys.argv) > 3 else None)
        elif cmd == "evaluate":
            evaluate()
        else:
            print("Usage:")
            print("  python train.py                    # train model")
            print("  python train.py predict image.jpg  # predict single image")
            print("  python train.py evaluate           # evaluate on val set")
    else:
        train()
