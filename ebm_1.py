import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from sentence_transformers import SentenceTransformer
from rapidfuzz.distance import Levenshtein
import numpy as np
import pandas as pd
import nltk
import random

nltk.download("punkt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using: {DEVICE}")

# =====================================
# 1) Load Frozen LLaMA (Base LM)
# =====================================

LM_NAME = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(LM_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lm = AutoModelForCausalLM.from_pretrained(LM_NAME).to(DEVICE)
lm.resize_token_embeddings(len(tokenizer))
lm.eval()

for p in lm.parameters():
    p.requires_grad = False

hidden_dim = lm.config.hidden_size


# =====================================
# 2) Load CISR Style Encoder (Style Expert E2)
# =====================================

cisr_tokenizer = AutoTokenizer.from_pretrained("AnnaWegmann/Style-Embedding")

if cisr_tokenizer.pad_token is None:
    cisr_tokenizer.pad_token = cisr_tokenizer.eos_token

cisr = AutoModel.from_pretrained("AnnaWegmann/Style-Embedding").to(DEVICE)
cisr.eval()

def get_cisr_embedding(text):
    sents = nltk.sent_tokenize(text)
    sents = [s for s in sents if len(s.split()) > 3][:5]

    if len(sents) == 0:
        sents = [text]

    inputs = cisr_tokenizer(
        sents,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        out = cisr(**inputs).last_hidden_state
        emb = out.mean(dim=1)
        emb = F.normalize(emb, p=2, dim=1)
        return emb.mean(dim=0, keepdim=True)


# =====================================
# 3) Load Semantic Model (Meaning Expert E3)
# =====================================

semantic_model = SentenceTransformer("all-mpnet-base-v2").to(DEVICE)
semantic_model.eval()


# =====================================
# 4) Load Future Regressor (Style LM Expert E1)
# =====================================

class FutureRegressor(nn.Module):
    def __init__(self, hidden_dim, style_dim=768):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, style_dim)
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, style_dim),
            nn.Softplus()
        )

    def forward(self, z):
        return self.mu_head(z), self.sigma_head(z)

regressor = FutureRegressor(hidden_dim).to(DEVICE)
regressor.load_state_dict(torch.load("future_regressor.pt", map_location=DEVICE))
regressor.eval()


# =====================================
# 5) Load Proposal Model (T5 for MH Edits)
# =====================================

T5_NAME = "google/flan-t5-base"

t5_tokenizer = T5Tokenizer.from_pretrained(T5_NAME)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_NAME).to(DEVICE)
t5_model.eval()


# =====================================
# 6) Candidate Generation
# =====================================

def generate_candidates(input_text, k=5):

    prompt = (
        f"Rewrite the following sentence in different wording "
        f"while preserving meaning:\n\n"
        f"\"{input_text}\"\n\nRewritten:"
    )

    encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = lm.generate(
        **encoded,
        max_new_tokens=60,
        do_sample=True,
        temperature=1.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        num_return_sequences=k
    )

    texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    cleaned = [t.replace(prompt, "").strip() for t in texts]
    return cleaned


# =====================================
# 7) Scoring Function (E1–E4)
# =====================================

def score_candidate(candidate, input_text, target_style_vec):

    # ---- E2: style similarity ----
    cand_style = get_cisr_embedding(candidate)
    e2 = 1 - F.cosine_similarity(cand_style, target_style_vec).item()

    # ---- E3: semantic similarity ----
    emb_in = semantic_model.encode(input_text, normalize_embeddings=True)
    emb_out = semantic_model.encode(candidate, normalize_embeddings=True)
    e3 = 1 - float(np.dot(emb_in, emb_out))

    # ---- E4: edit distance penalty ----
    e4 = Levenshtein.distance(candidate, input_text) / max(len(input_text), 1)

    # ---- E1: future regressor likelihood ----
    tokens = tokenizer(candidate, return_tensors="pt", truncation=True).to(DEVICE)

    with torch.no_grad():
        out = lm(**tokens, output_hidden_states=True)
        last = out.hidden_states[-1][:, -1, :]
        mu, sigma = regressor(last)
        dist = torch.distributions.Normal(mu, sigma)
        e1 = -dist.log_prob(target_style_vec.to(DEVICE)).mean().item()

    copy_ratio = Levenshtein.normalized_similarity(candidate, input_text)

    return e1, e2, e3, e4, copy_ratio


def total_energy(scores):
    e1, e2, e3, e4, e_copy = scores
    return (
        1.0*e1 +
        2.0*e2 +
        2.0*e3 +
        1.0*e4 +
        3.0*e_copy
    )


# =====================================
# 8) Minimal MH Refinement (T5-based)
# =====================================

def propose_edit_t5(text):
    tokens = nltk.word_tokenize(text)

    if len(tokens) < 6:
        return text

    idx = random.randint(0, len(tokens) - 3)

    masked = tokens.copy()
    masked[idx:idx+2] = ["<extra_id_0>"]
    masked_text = " ".join(masked)

    inputs = t5_tokenizer(masked_text, return_tensors="pt").to(DEVICE)

    output = t5_model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=4
    )

    fill = t5_tokenizer.decode(output[0], skip_special_tokens=True)
    proposed = masked_text.replace("<extra_id_0>", fill).strip()

    return proposed


def mh_refine(text, target_style_vec, steps=15):
    current = text
    current_energy = total_energy(
        score_candidate(current, current, target_style_vec)
    )

    for _ in range(steps):
        proposal = propose_edit_t5(current)
        prop_energy = total_energy(
            score_candidate(proposal, current, target_style_vec)
        )

        if prop_energy < current_energy:
            current = proposal
            current_energy = prop_energy

    return current


# =====================================
# 9) MAIN PIPELINE
# =====================================

def rewrite_in_style(input_text, target_style_vec, k=5):
    candidates = generate_candidates(input_text, k)

    filtered = [
        c for c in candidates
        if Levenshtein.normalized_similarity(c, input_text) < 0.85
    ]
    if filtered:
        candidates = filtered

    scored = [(cand, total_energy(score_candidate(cand, input_text, target_style_vec)))
              for cand in candidates]

    best = min(scored, key=lambda x: x[1])[0]

    refined = mh_refine(best, target_style_vec, steps=15)
    return refined


# =====================================
# ✅ Example Usage
# =====================================

if __name__ == "__main__":
    vec = pd.read_parquet("author_style_vectors.parquet").iloc[0]["style_vector"]
    vec = torch.tensor(vec, dtype=torch.float).unsqueeze(0).to(DEVICE)

    input_text = "The project was completed successfully without delays."
    output = rewrite_in_style(input_text, vec, k=5)

    print("\n✅ Final Output:\n", output)
