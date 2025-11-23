import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

PARQUET_PATH = "blog_processed.parquet"
OUTPUT_PATH = "author_style_vectors.parquet"
MODEL_NAME = "AnnaWegmann/Style-Embedding"

# -------------------------
# Load model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

# -------------------------
# Load data safely
# -------------------------
df = pd.read_parquet(PARQUET_PATH)

# fix numpy arrays → list
df["sentences"] = df["sentences"].apply(
    lambda x: x.tolist() if isinstance(x, np.ndarray) else x
)

# ensure author_id is string
df["author_id"] = df["author_id"].astype(str)

# -------------------------
# Embedding helper
# -------------------------
def embed(sent_list):
    # ensure list[str]
    if isinstance(sent_list, np.ndarray):
        sent_list = sent_list.tolist()
    if isinstance(sent_list, str):
        sent_list = [sent_list]

    inputs = tokenizer(
        sent_list,
        padding=True,
        truncation=True,
        max_length=512,        # ✅ hard enforce limit
        return_tensors="pt"
    )

    # ✅ roberta doesn't use token_type_ids
    inputs.pop("token_type_ids", None)

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs).last_hidden_state
        emb = out.mean(dim=1)
        emb = F.normalize(emb, p=2, dim=1)

        return emb.mean(dim=0).cpu().numpy()

# -------------------------
# Compute vectors
# -------------------------
records = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing CISR vectors"):
    vec = embed(row["sentences"])
    records.append({
        "author_id": row["author_id"],
        "style_vector": vec
    })

# -------------------------
# Save output
# -------------------------
out = pd.DataFrame(records)
out.to_parquet(OUTPUT_PATH, index=False)

print(f"\n✅ Saved {len(out)} style vectors → {OUTPUT_PATH}")
