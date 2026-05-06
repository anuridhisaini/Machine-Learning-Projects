import faiss
import json
import numpy as np
from models.clip_encoder import CLIPEncoder

INDEX_PATH = "data/embeddings/index.faiss"
META_PATH  = "data/embeddings/metadata.json"

encoder  = CLIPEncoder()
index    = faiss.read_index(INDEX_PATH)

with open(META_PATH) as f:
    metadata = json.load(f)

def search(query_text: str, top_k: int = 3):
    print(f"\nQuery: '{query_text}'")
    print("-" * 40)
    query_vec = encoder.encode_text(query_text).reshape(1, -1).astype("float32")
    scores, indices = index.search(query_vec, top_k)

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        item = metadata[idx]
        print(f"  #{rank}  {item['label']:<35} score: {score:.3f}")

# Try different queries
search("ethnic outfit for a wedding")
search("casual everyday outfit")
search("formal office wear")
search("summer party dress")