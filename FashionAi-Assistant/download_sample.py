import os
import requests
from PIL import Image
from io import BytesIO

os.makedirs("data/catalog", exist_ok=True)

# Sample fashion image URLs (free to use)
sample_outfits = {
    "ethnic_lehenga_red.jpg":    "https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=400",
    "ethnic_saree_blue.jpg":     "https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?w=400",
    "casual_tshirt_white.jpg":   "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400",
    "casual_jeans_blue.jpg":     "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400",
    "formal_suit_black.jpg":     "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",
    "formal_blazer_grey.jpg":    "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=400",
    "wedding_sherwani_gold.jpg": "https://images.unsplash.com/photo-1619603364853-b636d3661703?w=400",
    "summer_dress_floral.jpg":   "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=400",
    "kurta_white_mens.jpg":      "https://images.unsplash.com/photo-1604600980034-a43976cfa456?w=400",
    "party_gown_black.jpg":      "https://images.unsplash.com/photo-1566174053879-31528523f8ae?w=400",
}

print("Downloading sample outfit images...")
for filename, url in sample_outfits.items():
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))
        img.save(f"data/catalog/{filename}")
        print(f"  ✓ {filename}")
    except Exception as e:
        print(f"  ✗ {filename} failed: {e}")

print(f"\nDone. {len(os.listdir('data/catalog'))} images saved to data/catalog/")