import faiss
import json
import numpy as np
from models.clip_encoder import CLIPEncoder
from models.nlp_parser import NLPParser

INDEX_PATH = "data/embeddings/index.faiss"
META_PATH  = "data/embeddings/metadata.json"

encoder = CLIPEncoder()
parser  = NLPParser()
index   = faiss.read_index(INDEX_PATH)

with open(META_PATH) as f:
    metadata = json.load(f)

def smart_search(user_text: str, top_k: int = 3):
    print(f"\nQuery: '{user_text}'")
    print("=" * 50)

    # Step 1 — NLP intent
    intent = parser.parse(user_text)
    print(f"  Understood: {intent['top_occasion']} · {intent['top_style']}")

    # Step 2 — Build enriched query
    # Combine user text + top intent labels for a richer CLIP query
    enriched = f"{user_text} {intent['top_occasion']} {intent['top_style']}"

    # Step 3 — CLIP encode
    query_vec = encoder.encode_text(enriched).reshape(1, -1).astype("float32")

    # Step 4 — FAISS search (get more candidates to rerank)
    scores, indices = index.search(query_vec, min(top_k * 3, len(metadata)))

    # Step 5 — Rerank by occasion keyword match
    top_occasion = intent["top_occasion"].split()[0].lower()  # e.g. "wedding"
    results = []
    for score, idx in zip(scores[0], indices[0]):
        item = metadata[idx]
        label = item["label"].lower()
        # Boost score if label contains the top occasion word
        boost = 0.05 if top_occasion in label else 0.0
        results.append((score + boost, item))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n  Results:")
    for rank, (score, item) in enumerate(results[:top_k], 1):
        print(f"    #{rank}  {item['label']:<35} score: {score:.3f}")

# ── Test it ───────────────────────────────────────────────────────────────────
smart_search("ethnic outfit for a wedding, not too heavy")
smart_search("casual everyday college outfit")
smart_search("formal office wear professional")
smart_search("light summer beach vacation dress")