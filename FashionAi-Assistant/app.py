import gradio as gr
import faiss, json, os
import numpy as np
from PIL import Image
import tempfile, cv2

from models.clip_encoder  import CLIPEncoder
from models.nlp_parser    import NLPParser
from models.body_analysis import BodyAnalyzer
from models.tryon import VirtualTryOn

# ── Load all models once at startup ──────────────────────────────────────────
print("Starting Fashion AI Assistant...")
encoder  = CLIPEncoder()
parser   = NLPParser()
analyzer = BodyAnalyzer()
index    = faiss.read_index("data/embeddings/index.faiss")

with open("data/embeddings/metadata.json") as f:
    metadata = json.load(f)

print("All models loaded. Launching app...")

# ── Core pipeline function ────────────────────────────────────────────────────
def fashion_pipeline(user_image, user_text):
    if user_image is None:
        return (
            "❌ Please upload a photo first.",
            "", "", "", "",
            [], []
        )
    if not user_text or not user_text.strip():
        return (
            "❌ Please describe what kind of outfit you want.",
            "", "", "", "",
            [], []
        )

    try:
        # Save uploaded image to temp file for OpenCV
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        Image.fromarray(user_image).save(tmp_path)

        # ── Module 1: Body Analysis ───────────────────────────────
        body   = analyzer.full_analysis(tmp_path)
        os.unlink(tmp_path)

        shape      = body["body_shape"]
        skin       = body["skin_tone"]
        color_hex  = body["detected_color"]
        palette    = body["color_palette"]
        tips       = body["style_tips"]
        height     = body["height_category"]

        body_summary = f"""**Body Shape:** {shape.title()}  
**Skin Tone:** {skin.title()} `{color_hex}`  
**Height:** {height.title()}  
**Wear:** {', '.join(tips['suits'][:3])}  
**Avoid:** {', '.join(tips['avoid'])}"""

        palette_display = "  ".join([f"`{c}`" for c in palette])

        # ── Module 2: NLP Intent ──────────────────────────────────
        intent = parser.parse(user_text)
        top_occ = intent["top_occasion"]
        top_sty = intent["top_style"]

        intent_summary = f"""**Occasion:** {top_occ.title()}  
**Style:** {top_sty.title()}  
**Season:** {list(intent['season'].keys())[0].title()}"""

        # ── Module 3: CLIP + FAISS + Reranking ───────────────────
        enriched = (
            f"{user_text} "
            f"{top_occ} {top_sty} "
            f"{shape} {skin}"
        )
        query_vec = encoder.encode_text(enriched).reshape(1, -1).astype("float32")
        scores, indices = index.search(query_vec, min(15, len(metadata)))

        top_occasion_word = top_occ.split()[0].lower()
        suited            = [s.split()[0].lower() for s in tips["suits"]]

        results = []
        for score, idx in zip(scores[0], indices[0]):
            item  = metadata[idx]
            label = item["label"].lower()
            boost = 0.0
            boost += 0.06 if top_occasion_word in label else 0.0
            boost += 0.04 if any(s in label for s in suited) else 0.0
            results.append((score + boost, item))

        results.sort(key=lambda x: x[0], reverse=True)
        top5 = results[:5]

        # Format recommendations as markdown
        rec_text = "## 👗 Your Personalized Outfit Recommendations\n\n"
        outfit_images = []

        for rank, (score, item) in enumerate(top5, 1):
            label    = item["label"].title()
            img_path = item["path"]
            rec_text += f"**#{rank} — {label}**  \n"
            rec_text += f"Match score: `{score:.3f}`  \n\n"

            if os.path.exists(img_path):
                outfit_images.append(img_path)

        # Shopping links
        top_outfit = top5[0][1]["label"].replace("_", " ") if top5 else user_text
        q = top_outfit.replace(" ", "+")
        shopping_md = f"""## 🛍️ Shop This Look

| Store | Link |
|-------|------|
| Myntra | [Search {top_outfit.title()}](https://www.myntra.com/{q}) |
| Meesho | [Search {top_outfit.title()}](https://www.meesho.com/search?q={q}) |
| Ajio | [Search {top_outfit.title()}](https://www.ajio.com/search/?text={q}) |
| Amazon | [Search {top_outfit.title()}](https://www.amazon.in/s?k={q}) |
| Flipkart | [Search {top_outfit.title()}](https://www.flipkart.com/search?q={q}) |
| Nykaa Fashion | [Search {top_outfit.title()}](https://www.nykaafashion.com/search?q={q}) |
"""

        return (
            body_summary,
            palette_display,
            intent_summary,
            rec_text,
            shopping_md,
            outfit_images,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (str(e), "", "", "", "", [])


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Atieler AI",
    theme=gr.themes.Base(primary_hue="pink", secondary_hue="purple", neutral_hue="slate"),
    css="""
    .gradio-container { max-width: 1200px !important; margin: auto; }
    h1 { text-align: center; font-size: 2.2rem !important; }
    .subtitle { text-align: center; color: #888; margin-bottom: 20px; }
    footer { display: none !important; }
    """
) as demo:

    gr.Markdown("# ✦ Atieler AI")
    gr.Markdown("<p class='subtitle'>Upload your photo · Describe your style · Get AI recommendations + Virtual Try-On</p>")

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════
        # TAB 1 — Recommendations
        # ══════════════════════════════════════════════════════════
        with gr.TabItem("👗 Get Recommendations"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📸 Step 1 — Your Photo")
                    image_input = gr.Image(
                        label="Upload your photo (full body works best)",
                        type="numpy", height=320,
                    )
                    gr.Markdown("### 💬 Step 2 — Describe Your Style")
                    text_input = gr.Textbox(
                        label="What are you looking for?",
                        placeholder='e.g. "ethnic outfit for a wedding, not too heavy"',
                        lines=2,
                    )
                    gr.Examples(
                        examples=[
                            ["ethnic outfit for a wedding, not too heavy"],
                            ["casual everyday college outfit"],
                            ["formal office wear professional"],
                            ["light summer beach vacation dress"],
                            ["party outfit for New Year eve"],
                        ],
                        inputs=text_input,
                        label="Quick examples",
                    )
                    submit_btn = gr.Button("✨ Get My Outfit Recommendations", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### 🔬 Body Analysis")
                    body_out    = gr.Markdown("_Upload a photo and click the button_")
                    palette_out = gr.Markdown("")
                    gr.Markdown("### 🧠 Style Intent")
                    intent_out  = gr.Markdown("_Waiting for your style preference_")

            gr.Markdown("---")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👗 Recommended Outfits")
                    recs_out = gr.Markdown("_Recommendations appear here_")
                with gr.Column(scale=1):
                    gr.Markdown("### 🖼️ Outfit Preview")
                    gallery_out = gr.Gallery(
                        show_label=False, columns=2,
                        height=300, object_fit="cover",
                    )

            gr.Markdown("---")
            gr.Markdown("### 🛍️ Where to Buy")
            shopping_out = gr.Markdown("_Shopping links appear after recommendations_")

        # ══════════════════════════════════════════════════════════
        # TAB 2 — Virtual Try-On
        # ══════════════════════════════════════════════════════════
        with gr.TabItem("✨ Virtual Try-On"):
            gr.Markdown("### Upload your photo + pick an outfit to try on")

            with gr.Row():
                with gr.Column(scale=1):
                    tryon_user_img = gr.Image(
                        label="📸 Your Photo",
                        type="numpy", height=320,
                    )
                    outfit_choices = gr.Dropdown(
                        choices=[f.replace(".jpg","").replace("_"," ").title()
                                 for f in sorted(os.listdir("data/catalog"))
                                 if f.endswith(".jpg")],
                        label="👗 Choose an outfit from catalog",
                        value=None,
                    )
                    tryon_btn = gr.Button("✨ Try It On!", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### Result — Side by Side Comparison")
                    tryon_out = gr.Image(
                        label="You | Try-On | Outfit",
                        type="numpy",
                        height=520,
                    )

            with gr.Row():
                tryon_status = gr.Markdown("_Select an outfit and click Try It On_")

    # ── Wire recommendation button ────────────────────────────────
    submit_btn.click(
        fn=fashion_pipeline,
        inputs=[image_input, text_input],
        outputs=[body_out, palette_out, intent_out, recs_out, shopping_out, gallery_out],
        show_progress=True,
    )

    # ── Wire try-on button ────────────────────────────────────────
    def run_tryon(user_img, outfit_choice):
        
        if user_img is None:
            return None, "❌ Please upload your photo first."
        if not outfit_choice:
            return None, "❌ Please select an outfit from the dropdown."
        try:
            outfit_file = outfit_choice.lower().replace(" ", "_") + ".jpg"
            outfit_path = os.path.join("data/catalog", outfit_file)

            if not os.path.exists(outfit_path):
                catalog_files = os.listdir("data/catalog")
                match = next((f for f in catalog_files
                            if outfit_choice.lower().replace(" ","_") in f), None)
                outfit_path = os.path.join("data/catalog", match) if match else None

            if not outfit_path:
                return None, f"❌ Outfit not found: {outfit_file}"

        # Get body info from analyzer for richer card
            import tempfile
            from PIL import Image as PILImage
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            PILImage.fromarray(user_img).save(tmp_path)
            body = analyzer.full_analysis(tmp_path)
            os.unlink(tmp_path)

            tryon_model = VirtualTryOn()
            tips        = body["style_tips"]["suits"]
            style_tip   = f"Try {tips[0]} and {tips[1]}." if len(tips) >= 2 else tips[0]

            result = tryon_model.try_on(
                user_img, outfit_path,
                outfit_name   = outfit_choice,
                body_shape    = body["body_shape"],
                skin_tone     = body["skin_tone"],
                style_tip     = style_tip,
            )

            if result is None:
                return None, "❌ Try-on failed."

            return result, f"✅ Style preview ready — **{outfit_choice}**"

        except Exception as e:
            import traceback; traceback.print_exc()
            return None, f"❌ Error: {str(e)}"

    tryon_btn.click(
        fn=run_tryon,
        inputs=[tryon_user_img, outfit_choices],
        outputs=[tryon_out, tryon_status],
        show_progress=True,
    )

    gr.Markdown(
        "<p style='text-align:center;color:#555;margin-top:20px;font-size:13px'>"
        "Built with CLIP · FAISS · BART · OpenCV · Gradio"
        "</p>"
    )

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )