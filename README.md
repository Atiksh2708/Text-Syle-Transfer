# Style-Controlled Text Rewriting using Future Regressor + EBM (with MH Refinement)

This repository implements a lightweight version of the STYLEMC framework for **style-controlled text rewriting**.  
Given 20–50 writing samples from a user, the system learns their style representation and rewrites any neutral text to match it — while preserving meaning and fluency.

**1. No fine-tuning of the base LLM  
**2. GPU-friendly (LLaMA-1B + flan-T5-base)  
**3. Style-aware scoring using Product-of-Experts  
**4. Metropolis–Hastings refinement for global consistency

---

##  How It Works

### **1) User Style Extraction**
- Uses **CISR style encoder**
- Computes a 768-dimensional averaged style vector from 20–50 example sentences

### **2) Candidate Generation**
- Frozen **LLaMA-3.2-1B** generates multiple paraphrase candidates

### **3) Energy-Based Scoring (E1–E4 Experts)**
| Expert | Purpose |
|-------|---------|
| **E1** | Future Regressor (style likelihood) |
| **E2** | CISR cosine similarity (style match) |
| **E3** | Semantic similarity via MPNet |
| **E4** | Edit-distance penalty |

### **4) Metropolis-Hastings Refinement**
- Uses **flan-T5-base** to propose small edits
- Accepts only lower-energy samples
- Produces final globally consistent rewrite

---

## Installation

# Model Weights

Download the trained model weights from [Google Drive](https://drive.google.com/file/d/1sUo-prHaGHVCgOcxeMvkAICpuj0JBKDO/view?usp=sharing) (22 MB)

Place the downloaded `future_regressor.pt` file in the root directory.

```bash
git clone https://github.com/Atiksh2708/Text-Syle-Transfer.git
cd yourrepo
python infer.py