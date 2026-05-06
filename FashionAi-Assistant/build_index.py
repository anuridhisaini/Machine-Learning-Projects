import os
import numpy as np
import faiss
import json
from models.clip_encoder import CLIPEncoder

CATALOG_DIR = "data/catalog"
INDEX_PATH  = "data/embeddings/index.faiss"
META_PATH   = "data/embeddings/metadata.json"

os.makedirs("data/embeddings", exist_ok=True)

encoder  = CLIPEncoder()
images   = sorted(os.listdir(CATALOG_DIR))

embeddings = []
metadata   = []

print(f"Encoding {len(images)} outfit images...\n")

for img_file in images:
    img_path = os.path.join(CATALOG_DIR, img_file)
    try:
        vec = encoder.encode_image(img_path)
        embeddings.append(vec)
        metadata.append({
            "filename": img_file,
            "path":     img_path,
            "label":    img_file.replace(".jpg", "").replace("_", " ")
        })
        print(f"  ✓ {img_file}")
    except Exception as e:
        print(f"  ✗ {img_file} — {e}")

# Stack into matrix and build FAISS index
matrix = np.array(embeddings, dtype="float32")
index  = faiss.IndexFlatIP(512)   # Inner product = cosine similarity (vectors are normalized)
index.add(matrix)

faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nIndex built — {index.ntotal} outfits indexed.")
print(f"Saved to: {INDEX_PATH}")
print(f"Metadata: {META_PATH}")