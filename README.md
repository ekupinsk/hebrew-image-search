# Hebrew Image Search

Search a collection of images by describing them in Hebrew, or by uploading a similar image.
The project compares several ways of making CLIP-style image retrieval work in Hebrew and measures them on a standard benchmark.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/ElyKok/hebrew-image-search) · **Data and embeddings:** [ElyKok/hebrew-clip-search-data](https://huggingface.co/datasets/ElyKok/hebrew-clip-search-data)

<!-- Add a GIF of the demo here: it is the first thing a reviewer will look at. -->

## How it works

1. Every image is encoded once into a vector with a CLIP image encoder, and the vectors are stored.
2. At query time, the Hebrew text is encoded into the same vector space.
3. The images with the highest cosine similarity to the query are returned.

No model is trained. The work is in choosing how the Hebrew text reaches CLIP's space, and measuring which choice works best.

## Methods compared

| Method | How Hebrew is handled | Image encoder |
|---|---|---|
| Translate → CLIP | Hebrew is machine-translated to English (`opus-mt-tc-big-he-en`), then encoded by English CLIP | ViT-B/32 and ViT-L/14 |
| Multilingual CLIP | Hebrew is encoded directly by `clip-ViT-B-32-multilingual-v1` | ViT-B/32 |
| HebCLIP | Hebrew is encoded directly by NVIDIA's Hebrew-only text encoder | ViT-L/14 |
| English reference | Original English captions, no Hebrew involved: an upper bound | ViT-B/32 and ViT-L/14 |

Methods are compared only against others that share the same image encoder, so a difference in score reflects the text side and not a bigger vision model.

## Results

Text-to-image retrieval on the Flickr30k 1K test set (1,000 images, 5,000 captions). Recall@K is the percentage of queries where the correct image appears in the top K results.

| Method | Image encoder | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| English reference | B/32 | 58.78 | 83.42 | 90.06 |
| Hebrew → translate → CLIP | B/32 | 54.62 | 79.42 | 87.02 |
| Hebrew direct, multilingual CLIP | B/32 | 30.40 | 55.68 | 66.52 |
| English reference | L/14 | 64.58 | 87.08 | 92.08 |
| Hebrew → translate → CLIP | L/14 | **60.74** | **83.40** | **89.58** |
| Hebrew direct, NVIDIA HebCLIP | L/14 | TBD | TBD | TBD |

### Findings

- **Translation clearly beats the multilingual model.** With the same image encoder, translate-then-CLIP finds the correct image first 54.6% of the time, versus 30.4% for the multilingual model.
- **Hebrew costs about 4 points.** With either image encoder, the translation pipeline lands roughly 4 points of R@1 below the English upper bound.
- **The larger image encoder helps everywhere.** ViT-L/14 adds about 6 points of R@1 over ViT-B/32, in English and in Hebrew alike.

### Limitations

The Hebrew queries were produced by machine-translating the English Flickr30k captions (`opus-mt-en-he`). This may favor the translate-back pipeline, since translating back partly recovers the original English sentence. A different model was used for the back-translation to reduce this effect. A sample of 100 captions is provided in `manual_review.csv` for rewriting into natural Hebrew, as a cleaner check.

## Project structure

| File | Purpose |
|---|---|
| `hebrew_clip_search_step1.ipynb` | Builds the dataset, translates captions, computes embeddings, runs the evaluation, uploads to Hugging Face |
| `hebclip_eval_cell.py` | Adds the HebCLIP row to the evaluation |
| `space/app.py` | The Gradio demo: text search and image-to-image search |
| `space/requirements.txt` | Dependencies for the demo |

## Running it

1. Open `hebrew_clip_search_step1.ipynb` in Google Colab with a T4 GPU and run all cells (about 20 minutes).
2. Create a Gradio Space with ZeroGPU hardware and upload the two files from `space/`.

## Tech stack

Python · PyTorch · Hugging Face Transformers · Sentence-Transformers · CLIP · MarianMT · Gradio · Hugging Face Hub and ZeroGPU
