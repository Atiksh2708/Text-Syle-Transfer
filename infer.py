
import re
import json
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import spacy
from textstat import textstat
from textblob import TextBlob
import requests
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

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

from rapidfuzz.distance import Levenshtein
import numpy as np
import nltk
import random

def start():

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
def sem():
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

def regr():
    regressor = FutureRegressor(hidden_dim).to(DEVICE)
    regressor.load_state_dict(torch.load("future_regressor.pt", map_location=DEVICE))
    regressor.eval()


# =====================================
# 5) Load Proposal Model (T5 for MH Edits)
# =====================================
def block():
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


# ============================================================================
# STYLE VECTOR ANALYZER
# ============================================================================

@dataclass
class StyleVector:
    """Comprehensive style profile with 100+ features"""
    
    # ===== LEXICAL FEATURES (required fields first) =====
    avg_word_length: float
    avg_sentence_length: float
    vocabulary_richness: float  # Type-token ratio
    rare_word_frequency: float
    hapax_legomena_ratio: float  # Words appearing exactly once
    
    # ===== SYNTACTIC FEATURES (required fields) =====
    avg_words_per_sentence: float
    complex_sentence_ratio: float
    question_ratio: float
    exclamation_ratio: float
    passive_voice_ratio: float
    
    # ===== SENTIMENT & EMOTION (required fields) =====
    sentiment_polarity: float
    sentiment_subjectivity: float
    
    # ===== READABILITY (required fields) =====
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    
    # ===== PART-OF-SPEECH (required fields) =====
    adjective_ratio: float
    adverb_ratio: float
    noun_ratio: float
    verb_ratio: float
    
    # ===== PUNCTUATION (required fields) =====
    punctuation_density: float
    
    # ===== OPTIONAL FIELDS WITH DEFAULTS =====
    # Lexical (optional)
    top_char_trigrams: List[str] = field(default_factory=list)
    top_char_fourgrams: List[str] = field(default_factory=list)
    intensifier_frequency: float = 0.0
    slang_frequency: float = 0.0
    
    # Syntactic (optional)
    sentence_length_std: float = 0.0
    simple_sentence_ratio: float = 0.0
    compound_sentence_ratio: float = 0.0
    avg_parse_tree_depth: float = 0.0
    max_parse_tree_depth: int = 0
    active_voice_ratio: float = 0.0
    present_tense_ratio: float = 0.0
    past_tense_ratio: float = 0.0
    future_tense_ratio: float = 0.0
    modal_verb_frequency: float = 0.0
    modal_verbs_used: List[str] = field(default_factory=list)
    
    # Sentiment & Emotion (optional)
    vader_positive: float = 0.0
    vader_negative: float = 0.0
    vader_neutral: float = 0.0
    vader_compound: float = 0.0
    top_emotions: Dict[str, float] = field(default_factory=dict)
    formality_score: float = 0.0
    politeness_score: float = 0.0
    concreteness_score: float = 0.0
    
    # Readability (optional)
    gunning_fog_index: float = 0.0
    coleman_liau_index: float = 0.0
    smog_index: float = 0.0
    sentence_length_variance: float = 0.0
    rhythm_score: float = 0.0
    
    # Part-of-Speech (optional)
    pronoun_ratio: float = 0.0
    determiner_ratio: float = 0.0
    preposition_ratio: float = 0.0
    
    # Punctuation (optional)
    comma_frequency: float = 0.0
    semicolon_frequency: float = 0.0
    colon_frequency: float = 0.0
    dash_frequency: float = 0.0
    ellipsis_frequency: float = 0.0
    parentheses_frequency: float = 0.0
    quote_frequency: float = 0.0
    exclamation_frequency: float = 0.0
    question_mark_frequency: float = 0.0
    all_caps_frequency: float = 0.0
    title_case_frequency: float = 0.0
    
    # Discourse & Coherence (optional)
    discourse_marker_frequency: float = 0.0
    discourse_markers_used: List[str] = field(default_factory=list)
    rhetorical_question_ratio: float = 0.0
    dialogue_ratio: float = 0.0
    narrative_score: float = 0.0
    analytical_score: float = 0.0
    
    # Common patterns (optional)
    top_words: List[str] = field(default_factory=list)
    top_bigrams: List[str] = field(default_factory=list)
    top_trigrams: List[str] = field(default_factory=list)
    archaic_words: List[str] = field(default_factory=list)
    formal_words: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)


class StyleAnalyzer:
    """Extracts comprehensive style features from text corpus"""
    
    def __init__(self):
        # Load spaCy for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize VADER for sentiment
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize emotion classifier (will lazy-load)
        self.emotion_classifier = None
        
        # Common archaic/formal word indicators
        self.archaic_indicators = [
            'thou', 'thee', 'thy', 'thine', 'ye', 'hath', 'doth',
            'ere', 'wherefore', 'whence', 'hence', 'thence',
            'betwixt', 'methinks', 'verily', 'forsooth', 'anon',
            'mayhap', 'perchance', 'hither', 'thither', 'yonder'
        ]
        
        self.formal_indicators = [
            'furthermore', 'moreover', 'nevertheless', 'consequently',
            'henceforth', 'heretofore', 'notwithstanding', 'wherein',
            'thereby', 'thus', 'hence', 'albeit', 'whilst', 'aforementioned',
            'pursuant', 'whereby', 'herein', 'thereof', 'herewith'
        ]
        
        # Intensifiers
        self.intensifiers = [
            'very', 'extremely', 'incredibly', 'absolutely', 'totally',
            'completely', 'utterly', 'quite', 'rather', 'so', 'too',
            'really', 'truly', 'highly', 'exceptionally', 'extraordinarily'
        ]
        
        # Common slang/colloquial expressions
        self.slang_indicators = [
            'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'yeah', 'nope',
            'yep', 'lol', 'omg', 'btw', 'fyi', 'idk', 'tbh', 'imo', 'imho'
        ]
        
        # Modal verbs
        self.modal_verbs = [
            'can', 'could', 'may', 'might', 'shall', 'should',
            'will', 'would', 'must', 'ought'
        ]
        
        # Discourse markers
        self.discourse_markers = [
            'however', 'therefore', 'furthermore', 'moreover', 'nevertheless',
            'nonetheless', 'meanwhile', 'consequently', 'thus', 'hence',
            'additionally', 'alternatively', 'conversely', 'likewise',
            'similarly', 'specifically', 'namely', 'indeed', 'naturally',
            'certainly', 'obviously', 'clearly', 'arguably', 'presumably'
        ]
        
        # Politeness markers
        self.politeness_markers = [
            'please', 'thank', 'thanks', 'grateful', 'appreciate',
            'kindly', 'excuse', 'pardon', 'sorry', 'apologize',
            'would you', 'could you', 'may i', 'if you don\'t mind'
        ]
    
    def _lazy_load_emotion_classifier(self):
        """Lazy load emotion classifier to save memory"""
        if self.emotion_classifier is None:
            try:
                print("Loading emotion classifier (first time only)...")
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None
                )
            except Exception as e:
                print(f"Warning: Could not load emotion classifier: {e}")
                self.emotion_classifier = None
        return self.emotion_classifier
    
    def _get_parse_tree_depth(self, token, current_depth=0):
        """Recursively calculate parse tree depth"""
        if not list(token.children):
            return current_depth
        return max([self._get_parse_tree_depth(child, current_depth + 1) 
                   for child in token.children])
    
    def _classify_sentence_type(self, sentence: str) -> str:
        """Classify sentence as simple, compound, or complex"""
        # Simple heuristic based on conjunctions and punctuation
        has_conjunction = any(word in sentence.lower() for word in ['and', 'but', 'or', 'nor', 'yet', 'so'])
        has_subordinate = any(word in sentence.lower() for word in ['because', 'although', 'if', 'when', 'while', 'since', 'unless'])
        has_comma_or_semicolon = ',' in sentence or ';' in sentence
        
        if has_subordinate or (has_comma_or_semicolon and has_conjunction):
            return 'complex'
        elif has_conjunction and has_comma_or_semicolon:
            return 'compound'
        else:
            return 'simple'
    
    def _get_char_ngrams(self, text: str, n: int) -> List[str]:
        """Extract character n-grams"""
        text_clean = re.sub(r'\s+', '', text.lower())
        ngrams = [text_clean[i:i+n] for i in range(len(text_clean)-n+1)]
        return [ng for ng, _ in Counter(ngrams).most_common(15)]
    
    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect emotions using transformer model"""
        classifier = self._lazy_load_emotion_classifier()
        if classifier is None:
            return {}
        
        try:
            # Sample text to avoid memory issues
            sample = text[:1000]
            results = classifier(sample)
            
            if results and len(results) > 0:
                emotion_scores = {item['label']: item['score'] for item in results[0]}
                # Return top 5 emotions
                sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_emotions[:5])
        except Exception as e:
            print(f"Warning: Emotion detection failed: {e}")
        
        return {}
    
    def _calculate_formality(self, doc, words: List[str]) -> float:
        """
        Calculate formality score based on:
        - Noun density
        - Formal word usage
        - Lack of contractions
        - Complex vocabulary
        """
        # Count nouns
        noun_count = sum(1 for token in doc if token.pos_ == 'NOUN')
        
        # Count formal words
        formal_count = sum(1 for w in words if w in self.formal_indicators)
        
        # Count contractions
        contractions = len(re.findall(r"\w+'\w+", " ".join(words)))
        
        # Formality score (0-1)
        score = 0.0
        score += min(noun_count / len(doc), 0.4) if len(doc) > 0 else 0  # Max 0.4
        score += min(formal_count / len(words), 0.3) * 10 if words else 0  # Max 0.3
        score += max(0, 0.3 - (contractions / len(words)) * 5) if words else 0  # Max 0.3
        
        return min(score, 1.0)
    
    def _calculate_concreteness(self, doc) -> float:
        """
        Calculate concreteness vs abstractness
        Concrete: higher noun/verb ratio, specific entities
        Abstract: higher adjective/adverb ratio, concepts
        """
        pos_counts = Counter([token.pos_ for token in doc])
        total = len(doc)
        
        if total == 0:
            return 0.5
        
        # Concrete indicators: nouns, proper nouns, numbers
        concrete_score = (pos_counts.get('NOUN', 0) + 
                         pos_counts.get('PROPN', 0) + 
                         pos_counts.get('NUM', 0)) / total
        
        # Abstract indicators: adjectives, adverbs
        abstract_score = (pos_counts.get('ADJ', 0) + 
                         pos_counts.get('ADV', 0)) / total
        
        # Return concreteness (0 = abstract, 1 = concrete)
        return min(concrete_score / (concrete_score + abstract_score + 0.1), 1.0)
    
    def _calculate_rhythm_score(self, sentence_lengths: List[int]) -> float:
        """
        Calculate rhythm/cadence score using coefficient of variation
        Low = monotone, High = dynamic/varied
        """
        if not sentence_lengths or len(sentence_lengths) < 2:
            return 0.0
        
        mean = np.mean(sentence_lengths)
        std = np.std(sentence_lengths)
        
        if mean == 0:
            return 0.0
        
        # Coefficient of variation
        cv = std / mean
        return min(cv, 2.0)  # Cap at 2.0 for normalization
    
    def analyze_corpus(self, texts: List[str]) -> StyleVector:
        """
        Analyze a corpus and extract comprehensive style vector with 100+ features
        
        Args:
            texts: List of sentences/paragraphs from the target author
        
        Returns:
            StyleVector with all style features
        """
        print(f"Analyzing {len(texts)} text samples...")
        
        # Combine for overall statistics
        full_text = " ".join(texts)
        
        # Tokenize
        words = re.findall(r'\b\w+\b', full_text.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', full_text) if s.strip()]
        
        # Process with spaCy for linguistic features
        print("Processing with spaCy...")
        doc = self.nlp(full_text[:1000000])  # Limit for memory
        
        # ========== LEXICAL FEATURES ==========
        print("Extracting lexical features...")
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Type-token ratio (vocabulary richness)
        unique_words = set(words)
        vocabulary_richness = len(unique_words) / len(words) if words else 0
        
        # Rare words (appearing only once)
        word_counts = Counter(words)
        rare_words = sum(1 for count in word_counts.values() if count == 1)
        rare_word_frequency = rare_words / len(words) if words else 0
        hapax_legomena_ratio = rare_words / len(unique_words) if unique_words else 0
        
        # Character n-grams
        top_char_trigrams = self._get_char_ngrams(full_text, 3)
        top_char_fourgrams = self._get_char_ngrams(full_text, 4)
        
        # Intensifiers and slang
        intensifier_count = sum(1 for w in words if w in self.intensifiers)
        intensifier_frequency = intensifier_count / len(words) if words else 0
        
        slang_count = sum(1 for w in words if w in self.slang_indicators)
        slang_frequency = slang_count / len(words) if words else 0
        
        # ========== SYNTACTIC FEATURES ==========
        print("Analyzing syntax...")
        sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        avg_words_per_sentence = np.mean(sentence_lengths) if sentence_lengths else 0
        sentence_length_std = np.std(sentence_lengths) if sentence_lengths else 0
        
        # Classify sentence types
        sentence_types = [self._classify_sentence_type(s) for s in sentences[:500]]  # Sample for speed
        type_counts = Counter(sentence_types)
        total_classified = len(sentence_types)
        
        simple_sentence_ratio = type_counts.get('simple', 0) / total_classified if total_classified else 0
        compound_sentence_ratio = type_counts.get('compound', 0) / total_classified if total_classified else 0
        complex_sentence_ratio = type_counts.get('complex', 0) / total_classified if total_classified else 0
        
        question_ratio = full_text.count('?') / len(sentences) if sentences else 0
        exclamation_ratio = full_text.count('!') / len(sentences) if sentences else 0
        
        # Parse tree depth
        parse_depths = []
        for sent in list(doc.sents)[:100]:  # Sample 100 sentences
            for token in sent:
                if token.dep_ == 'ROOT':
                    depth = self._get_parse_tree_depth(token)
                    parse_depths.append(depth)
                    break
        
        avg_parse_tree_depth = np.mean(parse_depths) if parse_depths else 0
        max_parse_tree_depth = max(parse_depths) if parse_depths else 0
        
        # Voice detection
        passive_markers = len(re.findall(r'\b(was|were|been|being)\s+\w+ed\b', full_text, re.IGNORECASE))
        passive_voice_ratio = passive_markers / len(sentences) if sentences else 0
        active_voice_ratio = 1.0 - min(passive_voice_ratio, 1.0)
        
        # Tense detection
        tense_list = []
        for token in doc:
            if token.pos_ == 'VERB':
                tense_val = token.morph.get('Tense')
                if tense_val:
                    # Convert list to string
                    tense_str = tense_val[0] if isinstance(tense_val, list) and tense_val else str(tense_val)
                    tense_list.append(tense_str)
        
        tense_counts = Counter(tense_list)
        total_verbs = sum(tense_counts.values())
        
        present_tense_ratio = sum(count for tense, count in tense_counts.items() if 'Pres' in str(tense)) / total_verbs if total_verbs else 0
        past_tense_ratio = sum(count for tense, count in tense_counts.items() if 'Past' in str(tense)) / total_verbs if total_verbs else 0
        future_tense_ratio = full_text.lower().count('will ') / len(words) if words else 0
        
        # Modal verbs
        modal_count = sum(1 for w in words if w in self.modal_verbs)
        modal_verb_frequency = modal_count / len(words) if words else 0
        modals_used = list(set([w for w in words if w in self.modal_verbs]))[:10]
        
        # ========== SENTIMENT & EMOTION ==========
        print("Analyzing sentiment and emotion...")
        blob = TextBlob(full_text[:5000])
        sentiment_polarity = blob.sentiment.polarity
        sentiment_subjectivity = blob.sentiment.subjectivity
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(full_text[:5000])
        vader_positive = vader_scores['pos']
        vader_negative = vader_scores['neg']
        vader_neutral = vader_scores['neu']
        vader_compound = vader_scores['compound']
        
        # Emotion detection (top emotions)
        top_emotions = self._detect_emotions(full_text)
        
        # Formality and other semantic features
        formality_score = self._calculate_formality(doc, words)
        
        # Politeness
        politeness_count = sum(1 for marker in self.politeness_markers 
                              if marker in full_text.lower())
        politeness_score = min(politeness_count / len(sentences), 1.0) if sentences else 0
        
        # Concreteness
        concreteness_score = self._calculate_concreteness(doc)
        
        # ========== READABILITY & RHYTHM ==========
        print("Calculating readability metrics...")
        sample_text = full_text[:5000]
        flesch_reading_ease = textstat.flesch_reading_ease(sample_text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(sample_text)
        gunning_fog_index = textstat.gunning_fog(sample_text)
        coleman_liau_index = textstat.coleman_liau_index(sample_text)
        smog_index = textstat.smog_index(sample_text)
        
        # Rhythm and cadence
        sentence_length_variance = np.var(sentence_lengths) if sentence_lengths else 0
        rhythm_score = self._calculate_rhythm_score(sentence_lengths)
        
        # ========== POS PATTERNS ==========
        print("Analyzing part-of-speech patterns...")
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)
        
        adjective_ratio = pos_counts.get('ADJ', 0) / total_tokens if total_tokens else 0
        adverb_ratio = pos_counts.get('ADV', 0) / total_tokens if total_tokens else 0
        noun_ratio = pos_counts.get('NOUN', 0) / total_tokens if total_tokens else 0
        verb_ratio = pos_counts.get('VERB', 0) / total_tokens if total_tokens else 0
        pronoun_ratio = pos_counts.get('PRON', 0) / total_tokens if total_tokens else 0
        determiner_ratio = pos_counts.get('DET', 0) / total_tokens if total_tokens else 0
        preposition_ratio = pos_counts.get('ADP', 0) / total_tokens if total_tokens else 0
        
        # ========== PUNCTUATION & FORMATTING ==========
        print("Analyzing punctuation patterns...")
        total_words = len(words)
        
        comma_frequency = full_text.count(',') / total_words if total_words else 0
        semicolon_frequency = full_text.count(';') / total_words if total_words else 0
        colon_frequency = full_text.count(':') / total_words if total_words else 0
        dash_frequency = (full_text.count('—') + full_text.count('–') + full_text.count(' - ')) / total_words if total_words else 0
        ellipsis_frequency = (full_text.count('...') + full_text.count('…')) / total_words if total_words else 0
        parentheses_frequency = full_text.count('(') / total_words if total_words else 0
        quote_frequency = (full_text.count('"') + full_text.count("'")) / total_words if total_words else 0
        exclamation_frequency = full_text.count('!') / total_words if total_words else 0
        question_mark_frequency = full_text.count('?') / total_words if total_words else 0
        
        # Overall punctuation density
        all_punct = re.findall(r'[,;:—–\-!?.()"\']', full_text)
        punctuation_density = len(all_punct) / total_words if total_words else 0
        
        # Capitalization patterns
        all_caps_words = re.findall(r'\b[A-Z]{2,}\b', full_text)
        all_caps_frequency = len(all_caps_words) / total_words if total_words else 0
        
        title_case_words = re.findall(r'\b[A-Z][a-z]+\b', full_text)
        title_case_frequency = len(title_case_words) / total_words if total_words else 0
        
        # ========== DISCOURSE & COHERENCE ==========
        print("Analyzing discourse features...")
        discourse_count = sum(1 for marker in self.discourse_markers 
                             if marker in full_text.lower())
        discourse_marker_frequency = discourse_count / len(sentences) if sentences else 0
        discourse_markers_used = list(set([marker for marker in self.discourse_markers 
                                           if marker in full_text.lower()]))[:15]
        
        # Rhetorical questions (questions that don't expect answers)
        questions = [s for s in sentences if '?' in s]
        # Simplified heuristic: questions with "why", "how could", "who knows" are often rhetorical
        rhetorical_indicators = ['why ', 'how could', 'who knows', 'what if', 'isn\'t it']
        rhetorical_questions = sum(1 for q in questions 
                                   if any(ind in q.lower() for ind in rhetorical_indicators))
        rhetorical_question_ratio = rhetorical_questions / len(sentences) if sentences else 0
        
        # Dialogue detection
        dialogue_count = full_text.count('"') / 2  # Approximate
        dialogue_ratio = min(dialogue_count / len(sentences), 1.0) if sentences else 0
        
        # Narrative vs analytical (simplified)
        narrative_indicators = ['said', 'told', 'walked', 'looked', 'felt', 'went', 'came', 'saw']
        analytical_indicators = ['therefore', 'thus', 'consequently', 'analysis', 'conclude', 'demonstrate']
        
        narrative_count = sum(1 for w in words if w in narrative_indicators)
        analytical_count = sum(1 for w in words if w in analytical_indicators)
        
        total_style_markers = narrative_count + analytical_count
        narrative_score = narrative_count / total_style_markers if total_style_markers else 0.5
        analytical_score = analytical_count / total_style_markers if total_style_markers else 0.5
        
        # ========== COMMON PATTERNS ==========
        print("Extracting common patterns...")
        # Top words (excluding stop words)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'that', 'which', 'this', 'have', 'upon', 'from', 'were', 'been', 'there', 'very',
            'is', 'it', 'as', 'be', 'are', 'was', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'when', 'can', 'had', 'by', 'not', 'your', 'but', 'said', 'each', 'which', 'do',
            'their', 'time', 'if', 'will', 'how', 'about', 'may', 'its', 'only', 'out', 'other',
            'also', 'after', 'use', 'two', 'how', 'then', 'first', 'any', 'these', 'new', 'no',
            'between', 'now', 'just', 'where', 'most', 'some', 'them', 'same', 'our', 'than',
            'into', 'has', 'look', 'before', 'even', 'much', 'own', 'want', 'him', 'way', 'find',
            'give', 'day', 'most', 'us'
        }
        content_words = [w for w in words if w not in stop_words and len(w) > 3]
        top_words = [word for word, _ in Counter(content_words).most_common(20)]
        
        # Bigrams and trigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        top_bigrams = [bg for bg, _ in Counter(bigrams).most_common(15)]
        
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
        top_trigrams = [tg for tg, _ in Counter(trigrams).most_common(10)]
        
        # Vocabulary preferences
        archaic_words = [w for w in unique_words if w in self.archaic_indicators]
        formal_words = [w for w in unique_words if w in self.formal_indicators]
        
        # ========== CREATE STYLE VECTOR ==========
        print("Compiling style vector...")
        style_vector = StyleVector(
            # Lexical
            avg_word_length=round(avg_word_length, 2),
            avg_sentence_length=round(avg_sentence_length, 2),
            vocabulary_richness=round(vocabulary_richness, 3),
            rare_word_frequency=round(rare_word_frequency, 3),
            hapax_legomena_ratio=round(hapax_legomena_ratio, 3),
            top_char_trigrams=top_char_trigrams[:10],
            top_char_fourgrams=top_char_fourgrams[:10],
            intensifier_frequency=round(intensifier_frequency, 4),
            slang_frequency=round(slang_frequency, 4),
            
            # Syntactic
            avg_words_per_sentence=round(avg_words_per_sentence, 2),
            sentence_length_std=round(sentence_length_std, 2),
            complex_sentence_ratio=round(complex_sentence_ratio, 3),
            simple_sentence_ratio=round(simple_sentence_ratio, 3),
            compound_sentence_ratio=round(compound_sentence_ratio, 3),
            question_ratio=round(question_ratio, 3),
            exclamation_ratio=round(exclamation_ratio, 3),
            avg_parse_tree_depth=round(avg_parse_tree_depth, 2),
            max_parse_tree_depth=max_parse_tree_depth,
            passive_voice_ratio=round(passive_voice_ratio, 3),
            active_voice_ratio=round(active_voice_ratio, 3),
            present_tense_ratio=round(present_tense_ratio, 3),
            past_tense_ratio=round(past_tense_ratio, 3),
            future_tense_ratio=round(future_tense_ratio, 3),
            modal_verb_frequency=round(modal_verb_frequency, 3),
            modal_verbs_used=modals_used,
            
            # Sentiment & Emotion
            sentiment_polarity=round(sentiment_polarity, 3),
            sentiment_subjectivity=round(sentiment_subjectivity, 3),
            vader_positive=round(vader_positive, 3),
            vader_negative=round(vader_negative, 3),
            vader_neutral=round(vader_neutral, 3),
            vader_compound=round(vader_compound, 3),
            top_emotions=top_emotions,
            formality_score=round(formality_score, 3),
            politeness_score=round(politeness_score, 3),
            concreteness_score=round(concreteness_score, 3),
            
            # Readability
            flesch_reading_ease=round(flesch_reading_ease, 2),
            flesch_kincaid_grade=round(flesch_kincaid_grade, 2),
            gunning_fog_index=round(gunning_fog_index, 2),
            coleman_liau_index=round(coleman_liau_index, 2),
            smog_index=round(smog_index, 2),
            sentence_length_variance=round(sentence_length_variance, 2),
            rhythm_score=round(rhythm_score, 3),
            
            # POS
            adjective_ratio=round(adjective_ratio, 3),
            adverb_ratio=round(adverb_ratio, 3),
            noun_ratio=round(noun_ratio, 3),
            verb_ratio=round(verb_ratio, 3),
            pronoun_ratio=round(pronoun_ratio, 3),
            determiner_ratio=round(determiner_ratio, 3),
            preposition_ratio=round(preposition_ratio, 3),
            
            # Punctuation
            punctuation_density=round(punctuation_density, 3),
            comma_frequency=round(comma_frequency, 3),
            semicolon_frequency=round(semicolon_frequency, 4),
            colon_frequency=round(colon_frequency, 4),
            dash_frequency=round(dash_frequency, 4),
            ellipsis_frequency=round(ellipsis_frequency, 4),
            parentheses_frequency=round(parentheses_frequency, 4),
            quote_frequency=round(quote_frequency, 3),
            exclamation_frequency=round(exclamation_frequency, 4),
            question_mark_frequency=round(question_mark_frequency, 4),
            all_caps_frequency=round(all_caps_frequency, 4),
            title_case_frequency=round(title_case_frequency, 3),
            
            # Discourse
            discourse_marker_frequency=round(discourse_marker_frequency, 3),
            discourse_markers_used=discourse_markers_used,
            rhetorical_question_ratio=round(rhetorical_question_ratio, 4),
            dialogue_ratio=round(dialogue_ratio, 3),
            narrative_score=round(narrative_score, 3),
            analytical_score=round(analytical_score, 3),
            
            # Common patterns
            top_words=top_words,
            top_bigrams=top_bigrams,
            top_trigrams=top_trigrams,
            archaic_words=archaic_words,
            formal_words=formal_words
        )
        
        print("[OK] Style analysis complete!")
        return style_vector
    
    def visualize_style(self, style_vector: StyleVector, author_name: str = "Author"):
        """Create comprehensive visualization of style features"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f"Comprehensive Style Profile: {author_name}", fontsize=18, fontweight='bold')
        
        # 1. Lexical features
        ax1 = axes[0, 0]
        features1 = ['Avg Word\nLength', 'Vocabulary\nRichness', 'Rare Word\nFreq', 'Hapax\nRatio']
        values1 = [
            style_vector.avg_word_length / 10,
            style_vector.vocabulary_richness,
            style_vector.rare_word_frequency * 10,
            style_vector.hapax_legomena_ratio
        ]
        ax1.bar(features1, values1, color=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
        ax1.set_title('Lexical Features', fontweight='bold')
        ax1.set_ylabel('Normalized Score')
        
        # 2. Sentence structure
        ax2 = axes[0, 1]
        sentence_types = ['Simple', 'Compound', 'Complex']
        sentence_values = [
            style_vector.simple_sentence_ratio,
            style_vector.compound_sentence_ratio,
            style_vector.complex_sentence_ratio
        ]
        ax2.pie(sentence_values, labels=sentence_types, autopct='%1.1f%%', 
                colors=['#a8e6cf', '#ffd3b6', '#ffaaa5'])
        ax2.set_title('Sentence Structure Distribution', fontweight='bold')
        
        # 3. Voice & Tense
        ax3 = axes[0, 2]
        features3 = ['Active\nVoice', 'Passive\nVoice', 'Present\nTense', 'Past\nTense']
        values3 = [
            style_vector.active_voice_ratio,
            style_vector.passive_voice_ratio,
            style_vector.present_tense_ratio,
            style_vector.past_tense_ratio
        ]
        ax3.bar(features3, values3, color=['#84fab0', '#8fd3f4', '#a6c1ee', '#fbc2eb'])
        ax3.set_title('Voice & Tense Patterns', fontweight='bold')
        ax3.set_ylabel('Ratio')
        
        # 4. POS distribution
        ax4 = axes[1, 0]
        pos_labels = ['Nouns', 'Verbs', 'Adj', 'Adv', 'Pron', 'Prep']
        pos_values = [
            style_vector.noun_ratio,
            style_vector.verb_ratio,
            style_vector.adjective_ratio,
            style_vector.adverb_ratio,
            style_vector.pronoun_ratio,
            style_vector.preposition_ratio
        ]
        ax4.pie(pos_values, labels=pos_labels, autopct='%1.1f%%', startangle=90,
                colors=['#fa709a', '#fee140', '#30cfd0', '#a8edea', '#fed6e3', '#c1dfc4'])
        ax4.set_title('Part-of-Speech Distribution', fontweight='bold')
        
        # 5. Readability metrics
        ax5 = axes[1, 1]
        readability_labels = ['Flesch\nEase', 'Grade\nLevel', 'Gunning\nFog', 'Coleman\nLiau']
        readability_values = [
            min(style_vector.flesch_reading_ease / 100, 1),
            min(style_vector.flesch_kincaid_grade / 20, 1),
            min(style_vector.gunning_fog_index / 20, 1),
            min(style_vector.coleman_liau_index / 20, 1)
        ]
        ax5.bar(readability_labels, readability_values, 
                color=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
        ax5.set_title('Readability Metrics', fontweight='bold')
        ax5.set_ylabel('Normalized Score')
        
        # 6. Sentiment & Emotion
        ax6 = axes[1, 2]
        sentiment_text = f"VADER Compound: {style_vector.vader_compound:.2f}\n"
        sentiment_text += f"Polarity: {style_vector.sentiment_polarity:.2f}\n"
        sentiment_text += f"Subjectivity: {style_vector.sentiment_subjectivity:.2f}\n\n"
        sentiment_text += "Top Emotions:\n"
        for emotion, score in list(style_vector.top_emotions.items())[:3]:
            sentiment_text += f"  {emotion}: {score:.2f}\n"
        ax6.text(0.1, 0.5, sentiment_text, fontsize=11, verticalalignment='center',
                family='monospace')
        ax6.axis('off')
        ax6.set_title('Sentiment & Emotion', fontweight='bold')
        
        # 7. Punctuation patterns
        ax7 = axes[2, 0]
        punct_features = ['Comma', 'Semi-\ncolon', 'Dash', 'Ellipsis', 'Quotes', 'Parens']
        punct_values = [
            style_vector.comma_frequency * 10,
            style_vector.semicolon_frequency * 100,
            style_vector.dash_frequency * 100,
            style_vector.ellipsis_frequency * 100,
            style_vector.quote_frequency * 10,
            style_vector.parentheses_frequency * 100
        ]
        ax7.bar(punct_features, punct_values, color=['#fa709a', '#fee140', '#30cfd0', '#a8edea', '#fed6e3', '#c1dfc4'])
        ax7.set_title('Punctuation Patterns', fontweight='bold')
        ax7.set_ylabel('Frequency (normalized)')
        
        # 8. Style characteristics
        ax8 = axes[2, 1]
        style_features = ['Formality', 'Politeness', 'Concrete-\nness', 'Rhythm\nVariation']
        style_values = [
            style_vector.formality_score,
            style_vector.politeness_score,
            style_vector.concreteness_score,
            min(style_vector.rhythm_score / 2, 1)
        ]
        ax8.bar(style_features, style_values, 
                color=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
        ax8.set_title('Style Characteristics', fontweight='bold')
        ax8.set_ylabel('Score (0-1)')
        
        # 9. Discourse & Narrative
        ax9 = axes[2, 2]
        stats_text = f"Parse Tree Depth: {style_vector.avg_parse_tree_depth:.1f}\n"
        stats_text += f"Sentence Length: {style_vector.avg_words_per_sentence:.1f} ± {style_vector.sentence_length_std:.1f}\n"
        stats_text += f"Modal Verb Freq: {style_vector.modal_verb_frequency:.3f}\n"
        stats_text += f"Discourse Markers: {style_vector.discourse_marker_frequency:.3f}\n"
        stats_text += f"Narrative Score: {style_vector.narrative_score:.2f}\n"
        stats_text += f"Analytical Score: {style_vector.analytical_score:.2f}\n"
        stats_text += f"Dialogue Ratio: {style_vector.dialogue_ratio:.2f}\n"
        stats_text += f"Intensifiers: {style_vector.intensifier_frequency:.4f}"
        ax9.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                family='monospace')
        ax9.axis('off')
        ax9.set_title('Discourse & Narrative Stats', fontweight='bold')
        
        plt.tight_layout()
        filename = f'style_profile_{author_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Comprehensive style profile saved as '{filename}'")
        plt.show()


# ============================================================================
# LLM-BASED STYLE TRANSFER
# ============================================================================

class StyleTransferEngine:
    """Uses style vector to guide LLM for style transfer"""

    def __init__(self, ollama_model: str = "gemma3:4b", concise_mode: bool = True):
        """
        Initialize the engine with the Ollama model name.

        Args:
            ollama_model: The name of the local Ollama model to use.
            concise_mode: If True, uses a more concise prompt to reduce over-elaboration.
        """
        self.ollama_model = ollama_model
        self.concise_mode = concise_mode

    def create_style_prompt(self, style_vector: StyleVector, author_name: str = "the target author", original_text: str = "") -> str:
        """
        Create a comprehensive style prompt based on the analyzed style vector.

        Args:
            style_vector: The style vector containing stylistic features.
            author_name: The name of the author whose style is being emulated.
            original_text: The original text to be transformed (for length constraints).

        Returns:
            A detailed prompt to guide the LLM in style transfer.
        """
        # Build comprehensive style characteristics
        emotion_str = ""
        if style_vector.top_emotions:
            top_3_emotions = list(style_vector.top_emotions.items())[:3]
            emotion_str = ", ".join([f"{emotion} ({score:.2f})" for emotion, score in top_3_emotions])
        
        # Calculate target word count based on original and style ratio
        original_word_count = len(original_text.split()) if original_text else 10
        # Allow 50% expansion for stylistic elements, but cap it
        max_expansion_ratio = 1.5
        target_max_words = int(original_word_count * max_expansion_ratio)
        
        prompt = f"""You are an expert literary style transfer assistant. Your task is to rewrite text to match the distinctive writing style of {author_name}, while preserving the original meaning and content completely.

CRITICAL RULES - FOLLOW STRICTLY:
1. Keep the EXACT same meaning, events, and information as the original
2. Do NOT add new ideas, events, or elaborations beyond what's in the original
3. Do NOT remove any important information from the original
4. ONLY change HOW things are expressed, not WHAT is expressed
5. Maintain similar LENGTH: Original has ~{original_word_count} words, your output should be {original_word_count}-{target_max_words} words MAX
6. If the original is SHORT (1-2 sentences), keep your output SHORT (1-2 sentences) in the new style
7. Output ONLY the rewritten text - no explanations, notes, or commentary

STYLE PROFILE OF {author_name.upper()}:
Based on computational analysis of {author_name}'s writing, here are the key stylistic patterns to emulate:

═══ SENTENCE STRUCTURE ═══
• Average sentence length: {style_vector.avg_words_per_sentence:.1f} words (variation: ±{style_vector.sentence_length_std:.1f})
• Sentence complexity: {style_vector.simple_sentence_ratio:.0%} simple, {style_vector.compound_sentence_ratio:.0%} compound, {style_vector.complex_sentence_ratio:.0%} complex
• Parse tree depth: {style_vector.avg_parse_tree_depth:.1f} (syntactic complexity)
• Rhythm variation: {style_vector.rhythm_score:.2f} (how varied sentence lengths are)

═══ WORD CHOICE & VOCABULARY ═══
• Average word length: {style_vector.avg_word_length:.1f} characters
• Vocabulary richness: {style_vector.vocabulary_richness:.3f} (unique words / total words)
• Preferred words: {', '.join(style_vector.top_words[:15])}
• Common phrases: {', '.join(style_vector.top_bigrams[:8])}
• Archaic/unusual words: {', '.join(style_vector.archaic_words[:5]) if style_vector.archaic_words else 'none'}
• Formal language: {', '.join(style_vector.formal_words[:5]) if style_vector.formal_words else 'minimal'}

═══ GRAMMAR & VOICE ═══
• Voice preference: {style_vector.active_voice_ratio:.0%} active, {style_vector.passive_voice_ratio:.0%} passive
• Tense distribution: {style_vector.present_tense_ratio:.0%} present, {style_vector.past_tense_ratio:.0%} past
• Modal verbs frequency: {style_vector.modal_verb_frequency:.3f} (use of can/could/would/should/etc.)
• Part-of-speech balance: {style_vector.noun_ratio:.1%} nouns, {style_vector.verb_ratio:.1%} verbs, {style_vector.adjective_ratio:.1%} adjectives, {style_vector.adverb_ratio:.1%} adverbs

═══ PUNCTUATION STYLE ═══
• Comma density: {style_vector.comma_frequency:.3f} per word
• Semicolon usage: {style_vector.semicolon_frequency:.4f} per word {'(frequent)' if style_vector.semicolon_frequency > 0.01 else '(rare)'}
• Dash usage: {style_vector.dash_frequency:.4f} per word {'(frequent)' if style_vector.dash_frequency > 0.01 else '(occasional)'}
• Exclamations: {style_vector.exclamation_frequency:.4f} per word
• Questions: {style_vector.question_mark_frequency:.4f} per word
• Ellipses: {style_vector.ellipsis_frequency:.4f} per word {'(uses suspense)' if style_vector.ellipsis_frequency > 0.005 else '(rare)'}

═══ TONE & EMOTION ═══
• Sentiment polarity: {style_vector.sentiment_polarity:.2f} (-1=negative, 0=neutral, +1=positive)
• Subjectivity: {style_vector.sentiment_subjectivity:.2f} (0=objective, 1=subjective)
• Emotional tone: {emotion_str if emotion_str else 'neutral'}
• Formality level: {style_vector.formality_score:.2f} (0=casual, 1=formal)
• Concreteness: {style_vector.concreteness_score:.2f} (0=abstract, 1=concrete)

═══ READABILITY ═══
• Reading ease: {style_vector.flesch_reading_ease:.1f} (0=very difficult, 100=very easy)
• Grade level: {style_vector.flesch_kincaid_grade:.1f} (years of education needed)
• Complexity index: {style_vector.gunning_fog_index:.1f}

═══ DISCOURSE PATTERNS ═══
• Discourse markers: {', '.join(style_vector.discourse_markers_used[:8]) if style_vector.discourse_markers_used else 'minimal'}
• Narrative vs analytical: {style_vector.narrative_score:.1%} narrative, {style_vector.analytical_score:.1%} analytical
• Dialogue usage: {style_vector.dialogue_ratio:.2f}

STYLE TRANSFER INSTRUCTIONS:
Match these metrics as closely as possible while keeping the output concise. Pay special attention to:
- Sentence length: Keep around {style_vector.avg_words_per_sentence:.0f} words per sentence
- Word choice from the preferred vocabulary: {', '.join(style_vector.top_words[:8])}
- Active/passive voice ratio: {style_vector.active_voice_ratio:.0%} active
- Punctuation patterns (especially {'semicolons, ' if style_vector.semicolon_frequency > 0.01 else ''}{'dashes, ' if style_vector.dash_frequency > 0.01 else ''}commas)
- Emotional tone: {style_vector.sentiment_polarity:.1f} sentiment
- Formality level: {style_vector.formality_score:.1f}

IMPORTANT: Match the LENGTH of the original. Do not expand unnecessarily!

Now, rewrite the following text in the style of {author_name}. Remember: preserve ALL content and meaning, maintain similar length (~{original_word_count} words), only change the expression:

TEXT TO REWRITE:
"""
        return prompt

    def transfer_with_ollama(self, text: str, style_vector: StyleVector, author_name: str = "the author") -> str:
        """
        Perform style transfer using the Ollama model.

        Args:
            text: The input text to be stylized.
            style_vector: The style vector containing stylistic features.
            author_name: The name of the author whose style is being emulated.

        Returns:
            The stylized text.
        """
        import subprocess as _subprocess
        import json as _json

        # Create the style prompt
        prompt = self.create_style_prompt(style_vector, author_name, text)
        full_prompt = f"{prompt}\n{text}"

        http_err = None

        # 1) Try Ollama HTTP API (if server is already running)
        try:
            api_url = "http://127.0.0.1:11434/api/generate"
            payload = {
                "model": self.ollama_model, 
                "prompt": full_prompt, 
                "stream": False,
                "options": {
                    "num_ctx": 2048,
                    "num_predict": min(512, len(text.split()) * 3),  # Limit based on input
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            resp = requests.post(api_url, json=payload, timeout=60)
            if resp.ok:
                data = resp.json()
                if isinstance(data, dict) and 'response' in data:
                    result = data['response'].strip()
                    
                    # Validate output length (shouldn't be > 2x original)
                    original_words = len(text.split())
                    result_words = len(result.split())
                    
                    # If output is excessively long, warn user
                    if result_words > original_words * 2.5:
                        result = f"[Note: Output may be over-elaborated ({result_words} vs {original_words} original words)]\n\n{result}"
                    
                    return result
                else:
                    return f"OLLAMA_HTTP_OK_BUT_UNEXPECTED_JSON: {_json.dumps(data)[:2000]}"
            else:
                http_err = f"HTTP {resp.status_code}: {resp.text}"
        except Exception as e:
            http_err = f"HTTP request failed: {e}"

        # 2) No CLI fallback since Ollama CLI doesn't support 'generate' command
        return f"Error: Ollama HTTP API failed ({http_err}). Ensure Ollama server is running on port 11434 and model '{self.ollama_model}' is available."

    def transfer(self, text: str, style_vector: StyleVector, author_name: str = "the author") -> str:
        """
        Wrapper for style transfer using Ollama.

        Args:
            text: The input text to be stylized.
            style_vector: The style vector containing stylistic features.
            author_name: The name of the author whose style is being emulated.

        Returns:
            The stylized text.
        """
        return self.transfer_with_ollama(text, style_vector, author_name)


# ============================================================================
# MAIN USAGE
# ============================================================================

def load_corpus(filepath: str) -> List[str]:
    """Load text corpus from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    print("="*80)
    print("COMPREHENSIVE STYLE VECTOR ANALYSIS & TRANSFER SYSTEM")
    print("Analyzing 100+ linguistic features across 7 major categories")
    print("="*80)

    # 1. Load corpus
    print("\n[1] Loading corpus...")
    corpus = load_corpus("data/style_train.txt")
    print(f"[OK] Loaded {len(corpus)} samples")

    # 2. Analyze style
    print("\n[2] Analyzing writing style with 100+ features...")
    print("    This may take a few minutes for comprehensive analysis...")
    analyzer = StyleAnalyzer()
    style_vector = analyzer.analyze_corpus(corpus)

    # 3. Display comprehensive results
    print("\n" + "="*80)
    print("COMPREHENSIVE STYLE ANALYSIS RESULTS")
    print("="*80)
    
    print("\n[LEXICAL FEATURES]:")
    print(f"  • Avg sentence length: {style_vector.avg_words_per_sentence:.1f} words (σ={style_vector.sentence_length_std:.1f})")
    print(f"  • Avg word length: {style_vector.avg_word_length:.2f} characters")
    print(f"  • Vocabulary richness: {style_vector.vocabulary_richness:.3f}")
    print(f"  • Hapax legomena ratio: {style_vector.hapax_legomena_ratio:.3f}")
    print(f"  • Intensifier frequency: {style_vector.intensifier_frequency:.4f}")
    print(f"  • Slang frequency: {style_vector.slang_frequency:.4f}")
    
    print("\n[SYNTACTIC FEATURES]:")
    print(f"  • Simple sentences: {style_vector.simple_sentence_ratio:.1%}")
    print(f"  • Compound sentences: {style_vector.compound_sentence_ratio:.1%}")
    print(f"  • Complex sentences: {style_vector.complex_sentence_ratio:.1%}")
    print(f"  • Avg parse tree depth: {style_vector.avg_parse_tree_depth:.2f}")
    print(f"  • Active voice: {style_vector.active_voice_ratio:.1%}")
    print(f"  • Passive voice: {style_vector.passive_voice_ratio:.1%}")
    print(f"  • Present tense: {style_vector.present_tense_ratio:.1%}")
    print(f"  • Past tense: {style_vector.past_tense_ratio:.1%}")
    print(f"  • Modal verb frequency: {style_vector.modal_verb_frequency:.3f}")
    
    print("\n[SENTIMENT & EMOTION]:")
    print(f"  • VADER compound: {style_vector.vader_compound:.3f}")
    print(f"  • Sentiment polarity: {style_vector.sentiment_polarity:.3f}")
    print(f"  • Sentiment subjectivity: {style_vector.sentiment_subjectivity:.3f}")
    if style_vector.top_emotions:
        print(f"  • Top emotions: {', '.join([f'{k}({v:.2f})' for k, v in list(style_vector.top_emotions.items())[:3]])}")
    
    print("\n📖 READABILITY:")
    print(f"  • Flesch Reading Ease: {style_vector.flesch_reading_ease:.1f}")
    print(f"  • Grade Level: {style_vector.flesch_kincaid_grade:.1f}")
    print(f"  • Gunning Fog Index: {style_vector.gunning_fog_index:.1f}")
    print(f"  • Coleman-Liau Index: {style_vector.coleman_liau_index:.1f}")
    print(f"  • SMOG Index: {style_vector.smog_index:.1f}")
    print(f"  • Rhythm score (variation): {style_vector.rhythm_score:.3f}")
    
    print("\n[STYLE CHARACTERISTICS]:")
    print(f"  • Formality: {style_vector.formality_score:.3f}")
    print(f"  • Politeness: {style_vector.politeness_score:.3f}")
    print(f"  • Concreteness: {style_vector.concreteness_score:.3f}")
    print(f"  • Narrative score: {style_vector.narrative_score:.3f}")
    print(f"  • Analytical score: {style_vector.analytical_score:.3f}")
    
    print("\n[PUNCTUATION]:")
    print(f"  • Comma frequency: {style_vector.comma_frequency:.3f}")
    print(f"  • Semicolon frequency: {style_vector.semicolon_frequency:.4f}")
    print(f"  • Dash frequency: {style_vector.dash_frequency:.4f}")
    print(f"  • Ellipsis frequency: {style_vector.ellipsis_frequency:.4f}")
    print(f"  • Quote frequency: {style_vector.quote_frequency:.3f}")
    
    print("\n[DISCOURSE]:")
    print(f"  • Discourse marker frequency: {style_vector.discourse_marker_frequency:.3f}")
    print(f"  • Rhetorical question ratio: {style_vector.rhetorical_question_ratio:.4f}")
    print(f"  • Dialogue ratio: {style_vector.dialogue_ratio:.3f}")
    if style_vector.discourse_markers_used:
        print(f"  • Common discourse markers: {', '.join(style_vector.discourse_markers_used[:5])}")
    
    print("\n[VOCABULARY PATTERNS]:")
    print(f"  • Top words: {', '.join(style_vector.top_words[:8])}")
    print(f"  • Top bigrams: {', '.join(style_vector.top_bigrams[:3])}")
    if style_vector.archaic_words:
        print(f"  • Archaic words: {', '.join(style_vector.archaic_words[:5])}")
    if style_vector.formal_words:
        print(f"  • Formal words: {', '.join(style_vector.formal_words[:5])}")

    # 4. Visualize
    print("\n[3] Creating comprehensive visualization...")
    analyzer.visualize_style(style_vector, "Edgar Allan Poe")

    # 5. Save style vector
    print("\n[4] Saving style vector...")
    os.makedirs('style_vectors', exist_ok=True)
    with open('style_vectors/edgar_allan_poe.json', 'w') as f:
        json.dump(style_vector.to_dict(), f, indent=2)
    print("[OK] Saved to style_vectors/edgar_allan_poe.json")
    
    # Print summary statistics
    vector_dict = style_vector.to_dict()
    numeric_features = sum(1 for v in vector_dict.values() if isinstance(v, (int, float)))
    list_features = sum(1 for v in vector_dict.values() if isinstance(v, (list, dict)))
    print(f"\n[STYLE VECTOR SUMMARY]:")
    print(f"  • Total features: {len(vector_dict)}")
    print(f"  • Numeric features: {numeric_features}")
    print(f"  • Pattern features: {list_features}")

    # 6. Test style transfer
    print("\n[5] Testing style transfer with Ollama...")
    engine = StyleTransferEngine()

    test_text = "I went to the mall. It was a nice day."
    result = engine.transfer(test_text, style_vector, "Edgar Allan Poe")

    print("\n" + "="*80)
    print("STYLE TRANSFER RESULT:")
    print("="*80)
    print(f"Original: {test_text}")
    print(f"\nStyled: {result}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Example 1: Analyze a corpus
analyzer = StyleAnalyzer()
style_vector = analyzer.analyze_corpus(my_texts)
analyzer.visualize_style(style_vector, "Author Name")

# Example 2: Create transfer prompt
engine = StyleTransferEngine()
prompt = engine.transfer("Your text here", style_vector, "Author Name")
# Then paste this prompt into ChatGPT/Claude

# Example 3: Use with OpenAI API
engine = StyleTransferEngine(api_key="your-key")
result = engine.transfer_with_openai("Your text", style_vector, "Author")
"""
