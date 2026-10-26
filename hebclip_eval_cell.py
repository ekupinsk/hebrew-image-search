# HebCLIP evaluation - paste into a new Colab cell and run.
# Works in a fresh session too: it downloads the captions and the L/14 image
# vectors from your dataset repo instead of relying on earlier variables.
import json
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoTokenizer, AutoModel

REPO_ID = "ElyKok/hebrew-clip-search-data"
device = "cuda" if torch.cuda.is_available() else "cpu"

def fetch(path):
    return hf_hub_download(REPO_ID, path, repo_type="dataset")

captions = json.load(open(fetch("captions.json"), encoding="utf-8"))
captions_he = [c["he"] for c in captions]
gt = np.array([c["img"] for c in captions])
img_l14 = np.load(fetch("embeddings/images_L14.npy"))  # same openai CLIP ViT-L/14 image encoder HebCLIP is aligned to

tok = AutoTokenizer.from_pretrained("nvidia/hebclip-model")
model = AutoModel.from_pretrained("nvidia/hebclip-model").to(device).eval()

def encode_hebclip(texts, bs=128):
    out = []
    for i in tqdm(range(0, len(texts), bs), desc="HebCLIP"):
        batch = tok(texts[i:i+bs], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        # Mean over real tokens only. For a single sentence this equals the
        # model card's plain mean; in a padded batch it ignores the padding.
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (hidden * mask).sum(1) / mask.sum(1)
        out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def recall_at_k(q, imgs, gt, ks=(1, 5, 10)):
    sims = torch.from_numpy(q) @ torch.from_numpy(imgs).T
    top = sims.topk(max(ks), dim=1).indices.numpy()
    hits = top == gt[:, None]
    return {f"R@{k}": round(hits[:, :k].any(axis=1).mean() * 100, 2) for k in ks}

q = encode_hebclip(captions_he)
r = recall_at_k(q, img_l14, gt)
print("HebCLIP (Hebrew direct, L/14):", r)

# Append the row to results.csv and upload it back
df = pd.read_csv(fetch("results.csv"))
df = df[df["method"] != "Hebrew direct, NVIDIA HebCLIP L/14"]
df = pd.concat([df, pd.DataFrame([{"method": "Hebrew direct, NVIDIA HebCLIP L/14", "image_encoder": "L14", **r}])],
               ignore_index=True)
df.to_csv("results.csv", index=False)
np.save("text_L14_Hebrew_direct_HebCLIP.npy", q)

api = HfApi()
api.upload_file(path_or_fileobj="results.csv", path_in_repo="results.csv",
                repo_id=REPO_ID, repo_type="dataset", commit_message="Add HebCLIP results")
api.upload_file(path_or_fileobj="text_L14_Hebrew_direct_HebCLIP.npy",
                path_in_repo="embeddings/text_L14_Hebrew_direct_HebCLIP.npy",
                repo_id=REPO_ID, repo_type="dataset", commit_message="Add HebCLIP text embeddings")
df
