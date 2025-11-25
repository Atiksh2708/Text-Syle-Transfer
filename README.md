# Articunet — AI-Powered Text Style Transfer ✨

Articunet is a hybrid text style transfer system that blends **Transformer-based style encoders** with **semantic attribute ensembles** (e.g., sentiment metrics, readability features).
A **frozen LLM** acts as the generation backbone, guided by a **learned regressor over content & style representations** — enabling controllable style rewriting while preserving meaning.

---

## 🚀 Features

✔ Hybrid Transformer + Ensemble style extraction
✔ Target author style learned from **10–20 input examples**
✔ Web UI (Flask app) + CLI inference ✓
✔ Fast inference using **Ollama** 
✔ No expensive fine-tuning required

---

## 📦 Installation

### 1️⃣ Clone Repository

```bash
cd Text-Style-Transfer
```
(current root directory)

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Ollama Setup (LLM Backend)

Articunet uses Ollama for inference.

Install Ollama:

**Linux / WSL**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**
Download: [https://ollama.com/download](https://ollama.com/download)

**macOS**

```bash
brew install ollama
```

> 🔹 If required, ensure Ollama is added to PATH or environment variables.

Then pull the required model:

```bash
ollama pull gemma3:4b
```

Start the Ollama service (if not auto-started):

```bash
ollama serve
```

---

## ▶️ Running the Application

### ✨ Full Web App Demo (Recommended)

```bash
python app.py
```

Then open in browser:

```
http://127.0.0.1:5002/
```

📌 **How to Use UI**

* Paste **10–20 target style texts** (e.g., works of the author you want to mimic)
* Click on **Analyze Style** button
* Enter your **test sentence** to rewrite
* Click **Transform Text**

---

### 💻 Terminal-Based Inference

```bash
python infer.py
```

📌 Put your texts here inside `infer.py`:

| Section               | Line No. | What to Edit                               |
| --------------------- | -------- | ------------------------------------------ |
| Target style examples | `~1331`  | Insert 10–20 sentences (newline-separated) |
| Test sentence         | `~1454`  | Insert one input sentence to rewrite       |

Run and get output instantly in the terminal.





