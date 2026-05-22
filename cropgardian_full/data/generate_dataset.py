"""
PlantVillage Dataset Generator
================================
Generates synthetic training images for all 38 PlantVillage classes.
Each class gets images with realistic HSV color profiles matching real
PlantVillage disease signatures.

Usage:
    python generate_dataset.py                  # generate all 38 classes
    python generate_dataset.py --samples 200    # 200 images per class
    python generate_dataset.py --quick          # 50 per class for quick test

Then train:
    cd .. && python model/train.py
"""

import os
import random
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

# ── PlantVillage 38-class definitions ─────────────────────────────────────────
# Each class has:
#   - folder_name : exact PlantVillage folder naming convention
#   - base_green  : (R,G,B) of healthy leaf tissue
#   - lesion_color: (R,G,B) of disease lesion/spot
#   - lesion_pct  : what % of image has lesions (0.0 = healthy)
#   - pattern     : spot | blight | rust | mildew | mosaic | curl | healthy
#   - img_count   : approximate real dataset count (for reference)

CLASSES = [
    # ── TOMATO ──────────────────────────────────────────────────────────────
    {"id":  1, "folder": "Tomato___Late_blight",
     "base_green":(55,90,40),  "lesion_color":(30,20,15),   "lesion_pct":0.45, "pattern":"blight",   "img_count":1909},
    {"id":  2, "folder": "Tomato___Early_blight",
     "base_green":(65,105,50), "lesion_color":(100,65,20),  "lesion_pct":0.30, "pattern":"spot",     "img_count":1000},
    {"id":  3, "folder": "Tomato___Bacterial_spot",
     "base_green":(70,115,55), "lesion_color":(80,70,15),   "lesion_pct":0.20, "pattern":"spot",     "img_count":2127},
    {"id":  4, "folder": "Tomato___Leaf_Mold",
     "base_green":(60,100,45), "lesion_color":(90,120,50),  "lesion_pct":0.25, "pattern":"mold",     "img_count":952},
    {"id":  5, "folder": "Tomato___Septoria_leaf_spot",
     "base_green":(65,108,52), "lesion_color":(200,195,185),"lesion_pct":0.18, "pattern":"spot",     "img_count":1771},
    {"id":  6, "folder": "Tomato___Spider_mites Two-spotted_spider_mite",
     "base_green":(85,110,45), "lesion_color":(160,130,60), "lesion_pct":0.22, "pattern":"stipple",  "img_count":1676},
    {"id":  7, "folder": "Tomato___Target_Spot",
     "base_green":(68,112,50), "lesion_color":(110,70,25),  "lesion_pct":0.28, "pattern":"target",   "img_count":1404},
    {"id":  8, "folder": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
     "base_green":(160,170,60),"lesion_color":(210,205,80), "lesion_pct":0.55, "pattern":"curl",     "img_count":5357},
    {"id":  9, "folder": "Tomato___Tomato_mosaic_virus",
     "base_green":(80,130,55), "lesion_color":(150,170,60), "lesion_pct":0.35, "pattern":"mosaic",   "img_count":373},
    {"id": 10, "folder": "Tomato___healthy",
     "base_green":(55,130,45), "lesion_color":(55,130,45),  "lesion_pct":0.0,  "pattern":"healthy",  "img_count":1591},

    # ── POTATO ──────────────────────────────────────────────────────────────
    {"id": 11, "folder": "Potato___Late_blight",
     "base_green":(50,85,38),  "lesion_color":(25,15,10),   "lesion_pct":0.50, "pattern":"blight",   "img_count":1000},
    {"id": 12, "folder": "Potato___Early_blight",
     "base_green":(60,100,44), "lesion_color":(95,62,18),   "lesion_pct":0.28, "pattern":"spot",     "img_count":1000},
    {"id": 13, "folder": "Potato___healthy",
     "base_green":(52,125,42), "lesion_color":(52,125,42),  "lesion_pct":0.0,  "pattern":"healthy",  "img_count":152},

    # ── CORN ────────────────────────────────────────────────────────────────
    {"id": 14, "folder": "Corn_(maize)___Northern_Leaf_Blight",
     "base_green":(58,108,40), "lesion_color":(165,145,90), "lesion_pct":0.35, "pattern":"blight",   "img_count":985},
    {"id": 15, "folder": "Corn_(maize)___Common_rust_",
     "base_green":(62,115,43), "lesion_color":(180,85,20),  "lesion_pct":0.28, "pattern":"rust",     "img_count":1192},
    {"id": 16, "folder": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
     "base_green":(60,110,41), "lesion_color":(155,145,120),"lesion_pct":0.32, "pattern":"spot",     "img_count":513},
    {"id": 17, "folder": "Corn_(maize)___healthy",
     "base_green":(50,120,38), "lesion_color":(50,120,38),  "lesion_pct":0.0,  "pattern":"healthy",  "img_count":1162},

    # ── GRAPE ───────────────────────────────────────────────────────────────
    {"id": 18, "folder": "Grape___Black_rot",
     "base_green":(55,95,42),  "lesion_color":(20,15,12),   "lesion_pct":0.40, "pattern":"blight",   "img_count":1180},
    {"id": 19, "folder": "Grape___Esca_(Black_Measles)",
     "base_green":(58,98,44),  "lesion_color":(140,110,60), "lesion_pct":0.30, "pattern":"mosaic",   "img_count":1383},
    {"id": 20, "folder": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
     "base_green":(60,102,46), "lesion_color":(120,85,30),  "lesion_pct":0.25, "pattern":"spot",     "img_count":1076},
    {"id": 21, "folder": "Grape___healthy",
     "base_green":(52,118,42), "lesion_color":(52,118,42),  "lesion_pct":0.0,  "pattern":"healthy",  "img_count":423},

    # ── APPLE ───────────────────────────────────────────────────────────────
    {"id": 22, "folder": "Apple___Apple_scab",
     "base_green":(62,108,48), "lesion_color":(85,90,35),   "lesion_pct":0.25, "pattern":"scab",     "img_count":2016},
    {"id": 23, "folder": "Apple___Black_rot",
     "base_green":(65,110,50), "lesion_color":(35,22,18),   "lesion_pct":0.30, "pattern":"blight",   "img_count":621},
    {"id": 24, "folder": "Apple___Cedar_apple_rust",
     "base_green":(68,112,52), "lesion_color":(200,110,15), "lesion_pct":0.22, "pattern":"rust",     "img_count":275},
    {"id": 25, "folder": "Apple___healthy",
     "base_green":(55,125,45), "lesion_color":(55,125,45),  "lesion_pct":0.0,  "pattern":"healthy",  "img_count":1645},

    # ── RICE ────────────────────────────────────────────────────────────────
    {"id": 26, "folder": "Rice___Blast",
     "base_green":(50,105,38), "lesion_color":(170,150,100),"lesion_pct":0.30, "pattern":"spot",     "img_count":800},
    {"id": 27, "folder": "Rice___healthy",
     "base_green":(48,118,36), "lesion_color":(48,118,36),  "lesion_pct":0.0,  "pattern":"healthy",  "img_count":800},

    # ── PEPPER ──────────────────────────────────────────────────────────────
    {"id": 28, "folder": "Pepper,_bell___Bacterial_spot",
     "base_green":(62,112,48), "lesion_color":(85,72,18),   "lesion_pct":0.20, "pattern":"spot",     "img_count":997},
    {"id": 29, "folder": "Pepper,_bell___healthy",
     "base_green":(55,120,44), "lesion_color":(55,120,44),  "lesion_pct":0.0,  "pattern":"healthy",  "img_count":1478},

    # ── PEACH ───────────────────────────────────────────────────────────────
    {"id": 30, "folder": "Peach___Bacterial_spot",
     "base_green":(65,108,50), "lesion_color":(95,68,22),   "lesion_pct":0.22, "pattern":"spot",     "img_count":2297},
    {"id": 31, "folder": "Peach___healthy",
     "base_green":(58,122,46), "lesion_color":(58,122,46),  "lesion_pct":0.0,  "pattern":"healthy",  "img_count":360},

    # ── STRAWBERRY ──────────────────────────────────────────────────────────
    {"id": 32, "folder": "Strawberry___Leaf_scorch",
     "base_green":(60,100,44), "lesion_color":(140,55,30),  "lesion_pct":0.38, "pattern":"scorch",   "img_count":1109},
    {"id": 33, "folder": "Strawberry___healthy",
     "base_green":(55,118,42), "lesion_color":(55,118,42),  "lesion_pct":0.0,  "pattern":"healthy",  "img_count":456},

    # ── CHERRY ──────────────────────────────────────────────────────────────
    {"id": 34, "folder": "Cherry_(including_sour)___Powdery_mildew",
     "base_green":(62,105,48), "lesion_color":(215,212,205),"lesion_pct":0.25, "pattern":"mildew",   "img_count":1052},
    {"id": 35, "folder": "Cherry_(including_sour)___healthy",
     "base_green":(56,120,44), "lesion_color":(56,120,44),  "lesion_pct":0.0,  "pattern":"healthy",  "img_count":854},

    # ── SOYBEAN ─────────────────────────────────────────────────────────────
    {"id": 36, "folder": "Soybean___healthy",
     "base_green":(54,118,42), "lesion_color":(54,118,42),  "lesion_pct":0.0,  "pattern":"healthy",  "img_count":5090},

    # ── SQUASH ──────────────────────────────────────────────────────────────
    {"id": 37, "folder": "Squash___Powdery_mildew",
     "base_green":(65,108,50), "lesion_color":(218,215,208),"lesion_pct":0.30, "pattern":"mildew",   "img_count":1835},

    # ── BLUEBERRY ───────────────────────────────────────────────────────────
    {"id": 38, "folder": "Blueberry___healthy",
     "base_green":(52,110,42), "lesion_color":(52,110,42),  "lesion_pct":0.0,  "pattern":"healthy",  "img_count":1502},
]


def noise(val, spread=12):
    """Add realistic pixel noise."""
    return max(0, min(255, val + random.randint(-spread, spread)))

def make_leaf_base(size, base_color):
    """Create a realistic leaf background with texture."""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    r, g, b = base_color
    for y in range(size):
        for x in range(size):
            # Vein pattern: lighter near centre vertically
            dist_centre = abs(x - size//2) / (size//2)
            vein_light = int(15 * (1 - dist_centre))
            pixel = (noise(r+vein_light,10), noise(g+vein_light,12), noise(b+vein_light,8))
            img.putpixel((x, y), pixel)
    return img

def add_spots(img, lesion_color, count, radius_range=(8, 28), pattern="spot"):
    """Add disease spots/lesions."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    lr, lg, lb = lesion_color

    for _ in range(count):
        cx = random.randint(20, w-20)
        cy = random.randint(20, h-20)
        radius = random.randint(*radius_range)

        if pattern in ("spot", "scab", "target", "stipple"):
            # Core spot
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if dx*dx + dy*dy <= radius*radius:
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < w and 0 <= ny < h:
                            fade = 1 - (dx*dx+dy*dy)/(radius*radius+1)
                            pr,pg,pb = img.getpixel((nx,ny))
                            blended = (int(pr*(1-fade)+noise(lr,8)*fade),
                                       int(pg*(1-fade)+noise(lg,8)*fade),
                                       int(pb*(1-fade)+noise(lb,8)*fade))
                            img.putpixel((nx,ny), blended)
            # Yellow halo for bacterial/septoria
            if pattern in ("spot","target"):
                halo_r = int(radius * 1.5)
                for dx in range(-halo_r, halo_r+1):
                    for dy in range(-halo_r, halo_r+1):
                        d2 = dx*dx+dy*dy
                        if radius*radius < d2 <= halo_r*halo_r:
                            nx, ny = cx+dx, cy+dy
                            if 0 <= nx < w and 0 <= ny < h:
                                pr,pg,pb = img.getpixel((nx,ny))
                                img.putpixel((nx,ny),(min(255,pr+30), min(255,pg+20), pb//2))

        elif pattern == "blight":
            # Irregular blotch
            for dx in range(-radius, radius+1):
                for dy in range(-int(radius*0.7), int(radius*0.7)+1):
                    if random.random() > 0.3 and abs(dx)+abs(dy) <= int(radius*1.2):
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < w and 0 <= ny < h:
                            img.putpixel((nx,ny),(noise(lr,12),noise(lg,12),noise(lb,10)))

        elif pattern in ("rust","scorch"):
            # Orange/red pustule cluster
            for _ in range(6):
                ox = cx + random.randint(-radius, radius)
                oy = cy + random.randint(-radius//2, radius//2)
                pr2 = random.randint(3, 8)
                for dx in range(-pr2, pr2+1):
                    for dy in range(-pr2, pr2+1):
                        if dx*dx+dy*dy <= pr2*pr2:
                            nx, ny = ox+dx, oy+dy
                            if 0 <= nx < w and 0 <= ny < h:
                                img.putpixel((nx,ny),(noise(lr,15),noise(lg,10),noise(lb,8)))

        elif pattern in ("mildew","mold"):
            # White/grey powder patches
            for _ in range(8):
                ox = cx + random.randint(-radius, radius)
                oy = cy + random.randint(-radius, radius)
                pr2 = random.randint(4, 12)
                for dx in range(-pr2, pr2+1):
                    for dy in range(-pr2, pr2+1):
                        if dx*dx+dy*dy <= pr2*pr2:
                            nx, ny = ox+dx, oy+dy
                            if 0 <= nx < w and 0 <= ny < h:
                                v = noise(210, 20)
                                img.putpixel((nx,ny),(v,v,v-10))

        elif pattern == "mosaic":
            # Irregular yellow-green patches
            for _ in range(10):
                ox = cx + random.randint(-radius, radius)
                oy = cy + random.randint(-radius, radius)
                pr2 = random.randint(5, 15)
                for dx in range(-pr2, pr2+1):
                    for dy in range(-pr2, pr2+1):
                        if dx*dx+dy*dy <= pr2*pr2:
                            nx, ny = ox+dx, oy+dy
                            if 0 <= nx < w and 0 <= ny < h:
                                pr,pg,pb = img.getpixel((nx,ny))
                                # alternating light/dark green
                                if (nx+ny)%2==0:
                                    img.putpixel((nx,ny),(noise(160,15),noise(175,15),noise(60,10)))
                                else:
                                    img.putpixel((nx,ny),(noise(50,10),noise(95,12),noise(40,8)))

        elif pattern == "curl":
            # Pale yellow wash over patches
            bx, by = cx - radius, cy - radius
            ex, ey = cx + radius, cy + radius
            for px in range(max(0,bx), min(w,ex)):
                for py in range(max(0,by), min(h,ey)):
                    img.putpixel((px,py),(noise(175,18),noise(178,18),noise(65,10)))

    return img

def generate_image(cls_info, size=224):
    """Generate a single synthetic leaf image for a given class."""
    bg = make_leaf_base(size, cls_info["base_green"])
    pattern = cls_info["pattern"]
    if pattern == "healthy":
        # Apply subtle blur for realistic texture
        bg = bg.filter(ImageFilter.GaussianBlur(radius=0.5))
        return bg

    lesion_pct = cls_info["lesion_pct"]
    lc = cls_info["lesion_color"]
    # Scale spot count with lesion %
    spot_count = max(3, int(lesion_pct * 40 + random.randint(-3, 5)))

    img = add_spots(bg, lc, spot_count, pattern=pattern)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    return img

def generate_dataset(output_dir, samples_per_class=100, seed=42):
    """Generate full synthetic PlantVillage-style dataset."""
    random.seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    class_names = []
    total = 0

    print(f"\n{'='*60}")
    print(f"CropGardian — PlantVillage Synthetic Dataset Generator")
    print(f"Output: {output_path}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Total classes: {len(CLASSES)}")
    print(f"Estimated total images: {len(CLASSES) * samples_per_class:,}")
    print(f"{'='*60}\n")

    for cls in CLASSES:
        folder = output_path / cls["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        class_names.append(cls["folder"])

        for i in range(samples_per_class):
            # Random augmentation seed per image
            img = generate_image(cls)

            # Random augmentations
            # Flip
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.7:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # Rotate
            angle = random.choice([0, 90, 180, 270]) + random.randint(-15, 15)
            img = img.rotate(angle, fillcolor=(40, 80, 35))
            # Slight brightness variation
            from PIL import ImageEnhance
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.80, 1.20))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))

            img_path = folder / f"{cls['folder']}_{i:04d}.jpg"
            img.save(img_path, "JPEG", quality=85)

        total += samples_per_class
        print(f"  ✓ [{cls['id']:2d}/38] {cls['folder']:<55} {samples_per_class} images")

    # Save class names
    json_path = Path(__file__).parent.parent / "model" / "class_names.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Dataset generated: {total:,} images across {len(CLASSES)} classes")
    print(f"📁 Location: {output_path}")
    print(f"📄 class_names.json saved to: {json_path}")
    print(f"\nNext step: cd backend && python model/train.py")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PlantVillage-style synthetic dataset")
    parser.add_argument("--samples", type=int, default=100, help="Images per class (default: 100, real dataset: ~2300)")
    parser.add_argument("--quick",   action="store_true",   help="Quick test: 50 images per class")
    parser.add_argument("--full",    action="store_true",   help="Full simulation: 500 images per class (mimics PlantVillage scale)")
    parser.add_argument("--output",  type=str, default="plantvillage", help="Output directory name")
    args = parser.parse_args()

    if args.quick:
        samples = 50
    elif args.full:
        samples = 500
    else:
        samples = args.samples

    out_dir = Path(__file__).parent / args.output
    generate_dataset(out_dir, samples_per_class=samples)
