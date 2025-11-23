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
import nltk
import random

nltk.download("punkt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using: {DEVICE}")
DEVICE_CISR = "cpu"  # CISR can run on CPU to save GPU memory
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

CISR_NAME = "AnnaWegmann/Style-Embedding"

cisr_tokenizer = AutoTokenizer.from_pretrained(CISR_NAME)
if cisr_tokenizer.pad_token is None:
    cisr_tokenizer.pad_token = cisr_tokenizer.eos_token

cisr = AutoModel.from_pretrained(CISR_NAME).to(DEVICE_CISR)
cisr.eval()


def get_cisr_embedding(text: str) -> torch.Tensor:
    """
    Compute CISR style embedding for a single text string.
    Returns tensor of shape [1, 768].
    """
    sents = nltk.sent_tokenize(text)
    sents = [s for s in sents if len(s.split()) > 3][:8]

    if len(sents) == 0:
        sents = [text]

    inputs = cisr_tokenizer(
        sents,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE_CISR)

    with torch.no_grad():
        out = cisr(**inputs).last_hidden_state
        emb = out.mean(dim=1)
        emb = F.normalize(emb, p=2, dim=1)
        return emb.mean(dim=0, keepdim=True).to(DEVICE)  # [1, 768]


def compute_user_style_vector(sentences) -> torch.Tensor:
    """
    sentences: list of 20–50 user sentences (their writing)
    Returns: [1, 768] style vector (CISR-averaged)
    """
    cleaned = [s.strip() for s in sentences if isinstance(s, str) and len(s.split()) > 3]
    cleaned = cleaned[:50]

    if len(cleaned) < 5:
        raise ValueError("Please provide at least ~5 meaningful sentences for style estimation.")

    inputs = cisr_tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE_CISR)

    with torch.no_grad():
        out = cisr(**inputs).last_hidden_state
        emb = out.mean(dim=1)
        emb = F.normalize(emb, p=2, dim=1)
        style_vec = emb.mean(dim=0, keepdim=True)
        style_vec = F.normalize(style_vec, p=2, dim=1)
        return style_vec.to(DEVICE)  # [1, 768]


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
# 6) Candidate Generation (from LLaMA)
# =====================================

def generate_candidates(input_text, k=5):
    prompt = (
        f"Rewrite the following text in different wording "
        f"while preserving meaning:\n\n"
        f"\"{input_text}\"\n\nRewritten:"
    )

    encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = lm.generate(
        **encoded,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=k
    )

    texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    cleaned = [t.replace(prompt, "").strip() for t in texts]
    return cleaned


# =====================================
# 7) Scoring Function (E1–E4)
# =====================================

def score_candidate(candidate, input_text, target_style_vec):
    """
    Returns (e1, e2, e3, e4, copy_ratio)
    """
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

    # copy ratio (high = too similar)
    copy_ratio = Levenshtein.normalized_similarity(candidate, input_text)

    return e1, e2, e3, e4, copy_ratio


def total_energy(scores):
    e1, e2, e3, e4, e_copy = scores
    return (
            1.0*e1 +
            3*e2 +   # was 2.0
            2.0*e3 +
            1.0*e4 +
            3.0*e_copy
    )


# =====================================
# 8) Minimal MH Refinement (T5-based)
# =====================================

def propose_edit_t5(text: str) -> str:
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
        num_beams=5
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
    """
    input_text: neutral text to rewrite
    target_style_vec: [1, 768] style tensor from compute_user_style_vector
    """
    target_style_vec = target_style_vec.to(DEVICE)

    candidates = generate_candidates(input_text, k)

    # remove near copies of original
    filtered = [
        c for c in candidates
        if Levenshtein.normalized_similarity(c, input_text) < 0.85
    ]
    if filtered:
        candidates = filtered

    scored = [
        (cand, total_energy(score_candidate(cand, input_text, target_style_vec)))
        for cand in candidates
    ]

    best = min(scored, key=lambda x: x[1])[0]

    refined = mh_refine(best, target_style_vec, steps=100)
    return refined


if __name__ == "__main__":
    # ----- 1) User provides 20–50 sentences in their style -----
    user_sentences = [
    "I met the client yesterday to review the progress we’ve made so far.",
    "They seemed quite happy with the results and only had a few minor suggestions.",
    "We decided to schedule another catch-up next month to discuss the next phase.",
    "Overall, communication has been smooth and the collaboration is going well.",
    "I'll send them a short summary email later today so everything is on record.",
    "I also walked them through a couple of alternative approaches we could take if timelines shift.",
    "They appreciated the transparency and said they’d get back to us once they review things internally.",
    "There was a brief discussion about resource allocation, but nothing that affects our current plan.",
    "I reassured them that any adjustments on our side would be communicated well in advance.",
    "They mentioned that their leadership team is eager to see the final prototype.",
    "We agreed that sharing interim updates every two weeks should help keep everything aligned.",
    "I made a note to refine the documentation before our next sync to avoid any confusion later.",
    "They also asked if we could prepare a small demo for their internal stakeholders.",
    "I told them that wouldn’t be a problem as long as we finalize the feature list beforehand.",
    "They were particularly impressed with the pace at which we’ve been moving.",
    "There was a quick conversation about expanding the scope slightly, but they said it’s still under consideration.",
    "I clarified a few technical points so we’re all on the same page going into the next sprint.",
    "They appreciated the level of detail we included in the last report.",
    "I let them know we’re keeping an eye on potential bottlenecks but don’t expect any major issues.",
    "They confirmed that budget approval for the next phase should come through in a couple of weeks.",
    "We talked briefly about long-term integration plans, though that’s something we’ll revisit later.",
    "I asked them to share any additional feedback once they test the latest build on their end.",
    "They said they’ll loop in another team member who has context on the final deployment.",
    "I mentioned that we’re refining a few workflows to make the user experience smoother.",
    "They seemed glad to hear that and encouraged us to keep iterating on those details.",
    "We wrapped up the meeting by outlining the immediate next steps for both sides.",
    "I updated our internal tracker right after the call so the team stays informed.",
    "They thanked us for being consistent with communication, which they said makes collaboration much easier.",
    "I’ll also prepare a quick slide deck before the next meeting to highlight key milestones.",
    "Overall, the client relationship feels strong and we’re on track for a solid delivery."
]


    print("🔹 Computing user style vector from few-shot samples...")
    user_style_vec = compute_user_style_vector(user_sentences)

    # ----- 2) Rewrite any neutral text into this style -----
    input_text = (
    "The quarterly financial review indicated stronger-than-expected revenue growth across key segments. "
    "Operating margins improved due to disciplined cost management and favorable market conditions. "
    "Investor sentiment remained positive as the firm outperformed its earnings guidance for the third consecutive quarter. "
    "However, the report highlighted emerging risks related to currency fluctuations and rising borrowing costs. "
    "The finance team recommended adjusting the portfolio strategy to maintain resilience in the upcoming fiscal cycle. "
    "Overall, the outlook remains cautiously optimistic given current economic indicators."
)


    print("\n🔹 Rewriting input in user's style...\n")
    output = rewrite_in_style(input_text, user_style_vec, k=10)

    print("✅ Final Output:\n", output)

    print("\n🔹 Computing CISR cosine similarity...")

    with torch.no_grad():
        out_vec = get_cisr_embedding(output)  # [1,768]
        style_sim = F.cosine_similarity(out_vec, user_style_vec).item()
        in_vec= get_cisr_embedding(input_text)  # [1,768]
        in_style_sim = F.cosine_similarity(in_vec, user_style_vec).item()

    print(f"🎯 Style similarity (CISR cosine): {style_sim:.4f}")
    print(f"🔹 Input style similarity (CISR cosine): {in_style_sim:.4f}")

        

    # ✅ quick interpretation
    if style_sim >= 0.85:
        print("✅ Excellent style match")
    elif style_sim >= 0.70:
        print("✅ Good but can be improved")
    else:
        print("⚠️ Weak match — consider tuning weights or increasing MH steps")