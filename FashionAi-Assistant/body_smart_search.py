import os, faiss, json
import numpy as np
from models.clip_encoder  import CLIPEncoder
from models.nlp_parser    import NLPParser
from models.body_analysis import BodyAnalyzer

INDEX_PATH = "data/embeddings/index.faiss"
META_PATH  = "data/embeddings/metadata.json"

encoder  = CLIPEncoder()
parser   = NLPParser()
analyzer = BodyAnalyzer()
index    = faiss.read_index(INDEX_PATH)

with open(META_PATH) as f:
    metadata = json.load(f)

def full_pipeline(image_path: str, user_text: str, top_k: int = 3):
    print(f"\n{'='*60}")
    print(f"Photo : {image_path}")
    print(f"Query : '{user_text}'")
    print(f"{'='*60}")

    # ── Module 1: Body Analysis ───────────────────────────────────
    body = analyzer.full_analysis(image_path)
    print(f"\n[Module 1 — Body Analysis]")
    print(f"  Shape     : {body['body_shape']}")
    print(f"  Skin tone : {body['skin_tone']}  ({body['detected_color']})")
    print(f"  Wear      : {', '.join(body['style_tips']['suits'][:2])}")

    # ── Module 2: NLP Intent ──────────────────────────────────────
    intent = parser.parse(user_text)
    print(f"\n[Module 2 — NLP Intent]")
    print(f"  Occasion  : {intent['top_occasion']}")
    print(f"  Style     : {intent['top_style']}")

    # ── Module 3: CLIP + FAISS + Reranking ───────────────────────
    enriched = (
        f"{user_text} "
        f"{intent['top_occasion']} {intent['top_style']} "
        f"{body['body_shape']} {body['skin_tone']}"
    )
    query_vec = encoder.encode_text(enriched).reshape(1, -1).astype("float32")
    scores, indices = index.search(query_vec, min(top_k * 4, len(metadata)))

    top_occasion  = intent["top_occasion"].split()[0].lower()
    suited_styles = [s.split()[0].lower() for s in body["style_tips"]["suits"]]

    results = []
    for score, idx in zip(scores[0], indices[0]):
        item  = metadata[idx]
        label = item["label"].lower()
        boost = 0.0
        boost += 0.06 if top_occasion in label else 0.0
        boost += 0.04 if any(s in label for s in suited_styles) else 0.0
        results.append((score + boost, item))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n[Final Recommendations]")
    for rank, (score, item) in enumerate(results[:top_k], 1):
        print(f"  #{rank}  {item['label']:<35} score: {score:.3f}")

    return results[:top_k]

# ── Test all 3 modules together ───────────────────────────────────────────────
test_image = "data/catalog/casual_tshirt_white.jpg"
full_pipeline(test_image, "ethnic outfit for a wedding")
full_pipeline(test_image, "casual college outfit")
full_pipeline(test_image, "formal office wear")

