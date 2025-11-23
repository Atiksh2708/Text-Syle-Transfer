import pandas as pd
import nltk
import re

nltk.download("punkt")

INPUT_CSV = r"C:\Users\Lenovo\OneDrive\文档\Desktop\NLP\blogtext.csv"   # <-- change to your file
OUTPUT_PARQUET = "blog_processed.parquet"

# load trimmed data
df = pd.read_csv(INPUT_CSV, header=None)

# assign column names (based on your screenshot)
df.columns = [
    "blogger_id", "gender", "age",
    "industry", "zodiac", "date", "text"
]

# clean text function
def clean_text(t):
    if not isinstance(t, str):
        return ""
    
    t = t.replace("urllink", " ")      # remove url placeholders
    t = re.sub(r"http\S+", " ", t)     # strip URLs
    t = re.sub(r"<.*?>", " ", t)       # remove html
    t = re.sub(r"\s+", " ", t)         # normalize spaces
    return t.strip()

df["text"] = df["text"].apply(clean_text)

author_records = []

SENTENCES_PER_AUTHOR = 50   # good target for CISR

for author_id, group in df.groupby("blogger_id"):
    
    # merge all posts for that author
    full_text = " ".join(group["text"].tolist())
    
    # tokenize into sentences
    sentences = nltk.sent_tokenize(full_text)
    
    # filter short ones
    sentences = [s for s in sentences if len(s.split()) > 5]
    
    if len(sentences) < SENTENCES_PER_AUTHOR:
        continue  # skip small authors
    
    # store only required count
    sentences = sentences[:SENTENCES_PER_AUTHOR]
    
    author_records.append({
        "author_id": author_id,
        "num_sentences": len(sentences),
        "sentences": sentences
    })

out_df = pd.DataFrame(author_records)
out_df["author_id"] = out_df["author_id"].astype(str)
out_df.to_parquet(OUTPUT_PARQUET, index=False)

print(f"✅ Saved {len(out_df)} authors → {OUTPUT_PARQUET}")
