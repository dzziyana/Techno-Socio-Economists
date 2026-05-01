"""
Verifiability classifier: trained on FEVER, applied to PHEME source tweets.

MOTIVATION
----------
Nielsen & McConville (2022) showed that cross-dataset claim-level signals
reliably detect misinformation across languages and platforms. We adapt the
idea to PHEME: train a lightweight classifier on FEVER's VERIFIABLE vs NOT
VERIFIABLE labels, then score each PHEME source tweet. The hypothesis is
that hard-to-verify claims generate faster engagement because readers cannot
suppress the urge to share via a quick mental fact-check — a mechanism
consistent with Vosoughi et al. (2018) who identified novelty (also hard to
verify) as the primary driver of faster false-news spread.

REFERENCES
----------
Nielsen, R. K. & McConville, R. (2022). MuMiN: A Large-Scale Multilingual
    Multimodal Fact-Checked Misinformation Social Network Dataset.
    Findings of the Association for Computational Linguistics (ACL 2022).

Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018).
    FEVER: a Large-scale Dataset for Fact Extraction and VERification.
    NAACL-HLT 2018.

Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false
    news online. Science, 359(6380), 1146–1151.

DOMAIN-SHIFT NOTE
-----------------
FEVER claims are formal Wikipedia-style sentences (~8 words). PHEME source
tweets are informal, emoji-heavy, often incomplete (~15 words). The
classifier will therefore be imperfect on PHEME text — treat scores as a
noisy proxy for verifiability, not a ground-truth label. We report AUC on
the FEVER dev set for transparency, and use the score only as a continuous
predictor in regression/correlation analyses rather than as a hard label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fever(path: str | Path, max_samples: int | None = None) -> tuple[list[str], list[int]]:
    """
    Load FEVER claims and binary verifiability labels.

    Returns (claims, labels) where:
      label = 1  →  VERIFIABLE   (claim can be confirmed or refuted)
      label = 0  →  NOT VERIFIABLE  (not enough information exists)
    """
    claims, labels = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            row = json.loads(line)
            if "claim" not in row or "verifiable" not in row:
                continue
            claims.append(row["claim"])
            labels.append(1 if row["verifiable"] == "VERIFIABLE" else 0)
    return claims, labels


# ---------------------------------------------------------------------------
# Classifier pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    TF-IDF (unigrams + bigrams, sublinear TF) + Logistic Regression.

    class_weight='balanced' corrects for the 3:1 VERIFIABLE skew in FEVER,
    producing better-calibrated probabilities when applied to PHEME where
    the skew is unknown.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_features=50_000,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])


def train(train_claims: list[str], train_labels: list[int]) -> Pipeline:
    """Fit and return a trained pipeline."""
    pipe = build_pipeline()
    pipe.fit(train_claims, train_labels)
    return pipe


def evaluate(
    pipe: Pipeline,
    dev_claims: list[str],
    dev_labels: list[int],
) -> dict:
    """
    Evaluate on a held-out FEVER split.
    Returns a dict with accuracy, AUC, and a full classification report.
    """
    preds = pipe.predict(dev_claims)
    probs = pipe.predict_proba(dev_claims)[:, 1]
    report = classification_report(
        dev_labels, preds,
        target_names=["NOT VERIFIABLE", "VERIFIABLE"],
        output_dict=True,
    )
    return {
        "accuracy": report["accuracy"],
        "auc": float(roc_auc_score(dev_labels, probs)),
        "report": report,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_texts(pipe: Pipeline, texts: Sequence[str]) -> np.ndarray:
    """
    Return P(VERIFIABLE) for each text. Range [0, 1].

    Interpretation:
      High score (→ 1.0)  =  claim reads like something that can be checked
      Low score  (→ 0.0)  =  claim is vague, ambiguous, or lacks enough
                               context to verify — our proxy for "unverifiable"

    Lower scores on PHEME tweets predict faster first replies if the
    verifiability-as-mechanism hypothesis holds.
    """
    cleaned = [str(t).strip() if t and str(t).strip() else "unknown" for t in texts]
    return pipe.predict_proba(cleaned)[:, 1]


def add_verifiability_score(
    tm: pd.DataFrame,
    pipe: Pipeline,
    text_col: str = "source_text",
    score_col: str = "verifiability_score",
) -> pd.DataFrame:
    """
    Convenience wrapper: add a verifiability_score column to a threads
    dataframe and return the enriched copy.
    """
    tm = tm.copy()
    tm[score_col] = score_texts(pipe, tm[text_col].tolist())
    return tm
