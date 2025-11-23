import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import json

DATA_PATH = "future_regressor_dataset.parquet"
LM_NAME = "meta-llama/Llama-3.2-1B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", DEVICE)


# -------------------------
# Load + Fix Data
# -------------------------
df = pd.read_parquet(DATA_PATH)

# ✅ Fix prefix_ids stored as bytes/strings
def fix_prefix(x):
    if isinstance(x, bytes):
        return json.loads(x.decode("utf-8"))
    if isinstance(x, str):
        return json.loads(x)
    return x

df["prefix_ids"] = df["prefix_ids"].apply(fix_prefix)
df["prefix_ids"] = df["prefix_ids"].apply(lambda seq: [int(t) for t in seq])

# ✅ Ensure style_vector is float list
df["style_vector"] = df["style_vector"].apply(lambda v: [float(x) for x in v])


# -------------------------
# Dataset
# -------------------------
class PrefixDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            torch.tensor(row["prefix_ids"], dtype=torch.long),
            torch.tensor(row["style_vector"], dtype=torch.float)
        )


# -------------------------
# Collate function (padding)
# -------------------------
def collate(batch):
    prefixes, targets = zip(*batch)
    lengths = [len(x) for x in prefixes]
    max_len = max(lengths)

    padded = [F.pad(x, (0, max_len - len(x))) for x in prefixes]

    return torch.stack(padded), torch.stack(targets)


dataset = PrefixDataset(df)
loader = DataLoader(dataset, batch_size=12, shuffle=True, collate_fn=collate)


# -------------------------
# Load frozen LLaMA
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(LM_NAME)
llama = AutoModel.from_pretrained(LM_NAME).to(DEVICE)
llama.eval()

for p in llama.parameters():
    p.requires_grad = False

hidden_dim = llama.config.hidden_size  # e.g., 2048


# -------------------------
# Future Regressor (μ + σ)
# -------------------------
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
        mu = self.mu_head(z)
        sigma = self.sigma_head(z)
        return mu, sigma


regressor = FutureRegressor(hidden_dim).to(DEVICE)

# ✅ Initialize sigma for stability
with torch.no_grad():
    regressor.sigma_head[-2].bias.fill_(0.5)

# ✅ Lower learning rate
optimizer = torch.optim.AdamW(regressor.parameters(), lr=1e-5)


# -------------------------
# ✅ Safer Gaussian NLL
# -------------------------
def gaussian_nll(mu, sigma, target, eps=1e-6):
    var = sigma.pow(2) + eps
    return 0.5 * (
        (target - mu).pow(2) / var +
        torch.log(var)
    ).mean()


# -------------------------
# Training loop
# -------------------------
EPOCHS = 3

for epoch in range(EPOCHS):
    epoch_loss = 0
    steps = 0

    for prefix_ids, targets in tqdm(loader, desc=f"Epoch {epoch+1}"):

        prefix_ids = prefix_ids.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.no_grad():
            out = llama(prefix_ids).last_hidden_state
            z = out[:, -1, :]  # final token embedding

        mu, sigma = regressor(z)

        loss = gaussian_nll(mu, sigma, targets)

        optimizer.zero_grad()
        loss.backward()

        # ✅ gradient clipping
        torch.nn.utils.clip_grad_norm_(regressor.parameters(), 1.0)

        optimizer.step()

        epoch_loss += loss.item()
        steps += 1

    print(f"✅ Epoch {epoch+1} avg loss = {epoch_loss / steps:.4f}")


# -------------------------
# Save model
# -------------------------
torch.save(regressor.state_dict(), "future_regressor.pt")
print("✅ Saved → future_regressor.pt")
