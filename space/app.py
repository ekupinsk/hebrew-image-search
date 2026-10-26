"""Hebrew image search over Flickr30k (1k test images).

Runs on Hugging Face ZeroGPU. Image embeddings are precomputed in the
dataset repo, so at query time we only encode the text and do a dot product.
"""
import inspect
import numpy as np
import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer

DATA_REPO = "ElyKok/hebrew-clip-search-data"
# Download into a folder inside the app directory: Gradio only serves files
# from the working directory or /tmp, not from the Hugging Face cache.
data_dir = snapshot_download(DATA_REPO, repo_type="dataset", local_dir="data")

img_emb = {
    "B32": np.load(f"{data_dir}/embeddings/images_B32.npy"),
    "L14": np.load(f"{data_dir}/embeddings/images_L14.npy"),
}
n_images = img_emb["B32"].shape[0]
image_paths = [f"{data_dir}/images/{i:04d}.jpg" for i in range(n_images)]

# Models are moved to "cuda" at load time; ZeroGPU attaches a GPU only
# while a function decorated with @spaces.GPU is running.
multilingual = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1", device="cuda")
clip_l14 = SentenceTransformer("clip-ViT-L-14", device="cuda")
MT_NAME = "Helsinki-NLP/opus-mt-tc-big-he-en"
mt_tok = MarianTokenizer.from_pretrained(MT_NAME)
mt_model = MarianMTModel.from_pretrained(MT_NAME).to("cuda").eval()

# First entry is the default: translate+L14 scored best on Flickr30k (R@1 60.7 vs 30.4)
METHODS = {
    "תרגום לאנגלית ואז CLIP L/14 (מדויק יותר)": "translate_l14",
    "עברית ישירה, CLIP רב-לשוני B/32": "multilingual",
}


@spaces.GPU(duration=15)
def encode_query(query: str, method: str):
    """Returns (normalized query vector, image-encoder key, translated text or None)."""
    if method == "multilingual":
        q = multilingual.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return q.astype(np.float32), "B32", None

    batch = mt_tok([query], return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        gen = mt_model.generate(**batch, max_length=128, num_beams=4)
    english = mt_tok.batch_decode(gen, skip_special_tokens=True)[0]
    q = clip_l14.encode([english], normalize_embeddings=True, convert_to_numpy=True)
    return q.astype(np.float32), "L14", english


def search(query: str, method_label: str, k: int):
    query = (query or "").strip()
    if not query:
        return [], ""
    q, enc, english = encode_query(query, METHODS[method_label])
    scores = (img_emb[enc] @ q.T).ravel()
    top = np.argsort(-scores)[: int(k)]
    gallery = [(image_paths[i], f"{scores[i]:.3f}") for i in top]
    note = f"תרגום: {english}" if english else ""
    return gallery, note


@spaces.GPU(duration=15)
def encode_image(image):
    # CLIP L/14 image encoder: same space as the precomputed L14 image vectors
    q = clip_l14.encode([image.convert("RGB")], normalize_embeddings=True, convert_to_numpy=True)
    return q.astype(np.float32)


def search_by_image(image, k: int):
    if image is None:
        return []
    q = encode_image(image)
    scores = (img_emb["L14"] @ q.T).ravel()
    top = np.argsort(-scores)[: int(k)]
    return [(image_paths[i], f"{scores[i]:.3f}") for i in top]


EXAMPLES = [
    ["כלב רץ על הדשא", list(METHODS)[0], 8],
    ["ילדים משחקים בחוף הים", list(METHODS)[1], 8],
    ["אדם רוכב על אופניים ברחוב", list(METHODS)[0], 8],
    ["שתי נשים מדברות בבית קפה", list(METHODS)[1], 8],
]

# Right-to-left page; the slider and number boxes stay left-to-right so they
# still move in the expected direction.
CSS = """
.gradio-container { direction: rtl; text-align: right; }
.gradio-container input[type=range], .gradio-container input[type=number] { direction: ltr; }
"""
# Gradio 6 takes css in launch(); older versions take it in Blocks().
css_in_launch = "css" in inspect.signature(gr.Blocks.launch).parameters
blocks_kwargs = {} if css_in_launch else {"css": CSS}
launch_kwargs = {"css": CSS} if css_in_launch else {}

with gr.Blocks(title="חיפוש תמונות בעברית", **blocks_kwargs) as demo:
    gr.Markdown(
        "# חיפוש תמונות בעברית\n"
        "חפשו בין 1,000 תמונות של Flickr30k לפי תיאור בעברית, או לפי תמונה דומה."
    )
    with gr.Tabs():
        with gr.Tab("חיפוש לפי טקסט"):
            with gr.Row():
                query = gr.Textbox(label="מה לחפש?", placeholder="למשל: כלב רץ על הדשא", rtl=True, scale=4)
                method = gr.Radio(list(METHODS), value=list(METHODS)[0], label="שיטה", scale=2)
                k = gr.Slider(4, 16, value=8, step=4, label="מספר תוצאות", scale=1)
            btn = gr.Button("חפש", variant="primary")
            note = gr.Markdown()
            gallery = gr.Gallery(label="תוצאות", columns=4, height="auto")

            btn.click(search, [query, method, k], [gallery, note])
            query.submit(search, [query, method, k], [gallery, note])
            gr.Examples(EXAMPLES, [query, method, k], [gallery, note], fn=search, cache_examples=False)

        with gr.Tab("חיפוש לפי תמונה"):
            with gr.Row():
                img_in = gr.Image(type="pil", label="העלו תמונה", scale=2)
                k_img = gr.Slider(4, 16, value=8, step=4, label="מספר תוצאות", scale=1)
            img_btn = gr.Button("מצא תמונות דומות", variant="primary")
            img_gallery = gr.Gallery(label="תמונות דומות", columns=4, height="auto")
            img_btn.click(search_by_image, [img_in, k_img], img_gallery)

demo.launch(show_error=True, **launch_kwargs)
