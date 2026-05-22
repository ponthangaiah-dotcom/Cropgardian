# CropGardian — Quick Setup Guide

## Step 1: Generate Dataset
```bash
cd backend/data
python generate_dataset.py --samples 200    # ~7,600 images (fast, ~5 min)
python generate_dataset.py --full           # ~19,000 images (better accuracy)
```

> **Real PlantVillage dataset (recommended for production):**
> ```bash
> kaggle datasets download -d emmarex/plantdisease
> unzip plantdisease.zip -d backend/data/plantvillage/
> ```

## Step 2: Install Dependencies
```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 3: Train Model
```bash
cd backend
python model/train.py
# Training time: ~2-4 hrs GPU | ~10-20 hrs CPU
# Best model auto-saved to: backend/model/cropgardian_resnet50.pth
```

## Step 4: Setup Environment
```bash
cp .env.example .env
# Edit .env and change SECRET_KEY and JWT_SECRET
```

## Step 5: Run Backend
```bash
cd backend
python app.py
# API: http://localhost:5000
```

## Step 6: Open Frontend
```bash
# Simply open frontend/index.html in your browser
# OR serve it:
cd frontend
python -m http.server 8080
# Open: http://localhost:8080
```

## Test Prediction
```bash
python model/train.py predict path/to/leaf_image.jpg
```
