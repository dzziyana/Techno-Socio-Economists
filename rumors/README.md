# PHEME Spread Dynamics — Group Project

Comparative spread analysis of rumours vs. facts during breaking-news events,
using the PHEME-9 dataset (Zubiaga et al., 2016).

## Research question

> **Do rumours spread faster and farther than the truth during breaking-news events — and why?**

Vosoughi et al. (*Science*, 2018) found that falsehoods spread "farther, faster,
deeper, and more broadly" than truth on Twitter. We test whether this holds during
breaking-news events specifically — when facts haven't been established and
unverified claims fill the vacuum — and we propose a verifiability mechanism to
explain *why* the speed effect exists.

---

## Key findings

| Finding | Direction | Effect size | Interpretation |
|---|---|---|---|
| **Speed: time to first reply** | rumour < non-rumour | Cliff's δ = −0.26 | Rumours attract replies significantly faster |
| **Speed: time to half-cascade** | rumour < non-rumour | Cliff's δ = −0.18 | Speed advantage persists through the cascade |
| **Reach: cascade size** | null | — | Rumours do not reach more users overall |
| **Structure: depth vs. size** | null | — | No distinct structural signature for rumours |
| **Verifiability → speed** | less verifiable → faster | Spearman ρ = 0.09 (p < 10⁻¹²) | Claim ambiguity predicts faster engagement |
| **True rumours** | least verifiable class | Cliff's δ = −0.22 vs non-rumour | At posting time, even true claims were linguistically unverifiable |

**Summary:** Rumours engage faster, not bigger. The speed advantage is small but
consistent across events (holds in 4 of 7 eligible events for time-to-first-reply).
The "farther and deeper" result from Vosoughi does not replicate here — cascade
sizes and shapes are comparable across veracity classes.

---

## Project layout

```
rumors/
├── 01_data_wrangling.ipynb        ← Phase 1: parse PHEME → tidy dataframes
├── 02_cascade_metrics.ipynb       ← Phase 2: speed, reach, structural metrics
├── 03_statistical_comparison.ipynb← Phase 3: Mann-Whitney, BH correction, stratified analysis
├── 04_presentation_figures.ipynb  ← Phase 4: all presentation figures (figs 0–11)
├── 05_verifiability.ipynb         ← Phase 5: FEVER classifier + mechanism analysis
├── figures.py                     ← figure helper functions (imported by notebooks)
├── verifiability.py               ← FEVER loader, TF-IDF+LR classifier, scoring
├── stats_comparison.py            ← statistical tests, consistency checks
├── cascade_metrics.py             ← NetworkX cascade metrics
│
├── figures/                       ← all output figures (PNG + PDF)
│   ├── fig0_class_distribution    ← data overview: thread counts by event × veracity
│   ├── fig1_hook                  ← cascade trees: one rumour vs one non-rumour
│   ├── fig2_speed_first_reply     ← CDFs of time-to-first-reply (the headline finding)
│   ├── fig2b_speed_half_cascade   ← CDFs of time-to-half-cascade
│   ├── fig3_consistency_*         ← per-event consistency bars
│   ├── fig3b_per_event_speed_ci   ← per-event medians with 95% bootstrap CIs
│   ├── fig4_reach_null            ← cascade size by veracity (null result)
│   ├── fig5_structure_null        ← depth vs size scatter (null result)
│   ├── fig6_methodology           ← pooled vs stratified comparison
│   ├── fig7_verifiability_by_veracity   ← verifiability score histograms by class
│   ├── fig8_verifiability_speed_quartiles ← quartile bar chart: score → speed
│   ├── fig8b_verifiability_within_class   ← scatter within nonrumour + unverified
│   ├── fig9_verifiability_violin  ← violin plot: score distributions with significance
│   ├── fig10_verifiability_score_vs_speed ← scatter: score vs log(speed), OLS trend
│   └── fig11_verifiability_mechanism_summary ← 2-panel mechanism slide
│
├── all-rnr-annotated-threads/     ← raw PHEME data (not tracked in git)
└── data/processed/                ← cleaned parquet files (not tracked in git)
```

---

## How to reproduce

### 1. Get the data

Download PHEME from Figshare (`PHEME dataset for Rumour Detection and Veracity
Classification`, Zubiaga et al.). Unzip so the path
`data/all-rnr-annotated-threads/charliehebdo-all-rnr-threads/...` exists.

For the verifiability analysis, download the FEVER dataset
(Thorne et al., 2018) and place `train.jsonl` and `shared_task_dev.jsonl`
in `project/fever/`.

### 2. Install dependencies

```bash
pip install pandas pyarrow networkx matplotlib seaborn scipy scikit-learn
```

### 3. Run all notebooks in order

```bash
cd ./rumors
jupyter nbconvert --to notebook --execute --inplace \
  01_data_wrangling.ipynb \
  02_cascade_metrics.ipynb \
  03_statistical_comparison.ipynb \
  05_verifiability.ipynb \
  04_presentation_figures.ipynb
```

> **Note:** `05_verifiability.ipynb` must run before `04_presentation_figures.ipynb`
> so that `threads_with_verifiability.parquet` exists when the figures notebook loads it.

Expected output on full PHEME-9: ~6,400 threads, ~100k tweets, ~95k edges,
11 figures in `figures/` (each as `.png` and `.pdf`).

---

## The verifiability mechanism (Phase 5)

We trained a TF-IDF + logistic regression classifier on FEVER (145k Wikipedia
fact-verification claims, Thorne et al. 2018) to score each PHEME source tweet
for how *linguistically verifiable* it is — whether a reader could quickly
check it against available knowledge.

**Classifier:** trained on FEVER train split, evaluated on FEVER dev split
(ROC-AUC ≥ 0.80, calibrated with `class_weight='balanced'`).

**Hypothesis:** claims that are hard to verify generate faster engagement because
readers cannot suppress sharing via a quick mental fact-check.

**Results:**
- All rumour classes score lower on P(verifiable) than non-rumours (p < 0.001)
- Least verifiable quartile gets first replies in **1.87 min** (median);
  most verifiable quartile: **2.55 min** — monotone increase across all 4 quartiles
- Spearman ρ = 0.09, p < 10⁻¹², n = 5,747 (small but reliable)
- Effect holds within non-rumour and unverified subgroups separately

**Caveat:** ρ = 0.09 is a weak effect. The ~40-second median difference is
statistically reliable but practically modest. The finding supports the
verifiability mechanism as a *contributing factor*, not a primary driver.

Inspired by Nielsen & McConville (2022), *MuMiN: A Large-Scale Multilingual
Multimodal Fact-Checked Misinformation Social Network Dataset*, Findings of ACL.

---

## Key design decisions

- **We trust `structure.json`, not `in_reply_to_status_id`.** The structure
  file is the canonical reply tree; some replies have deleted JSONs but the
  tree topology is preserved.
- **`n_tweets_in_structure` is our cascade size**, not `n_reactions`.
- **Veracity has four classes**: `true`, `false`, `unverified`, `nonrumour`.
  Don't collapse `unverified` — it has distinct dynamics.
- **Always stratify by event.** Pooled stats can be event-confounded (see fig6).
- **BH correction** applied across all pairwise tests within each metric family.

## Limitations

- PHEME is small (~6,400 threads, 9 events) and English-only.
- Reply trees, not full retweet networks — depth is conversational, not diffusional.
- Snapshot data — cascades may have continued after collection.
- Annotations are journalist-curated, not crowdsourced — high precision, limited recall.
- Verifiability classifier trained on Wikipedia sentences (FEVER); transfer to
  informal tweet language is imperfect. Scores are a proxy, not ground truth.

## References

- Zubiaga et al. (2016). Analysing How People Orient to and Spread Rumours in
  Social Media. *PLOS ONE*.
- Vosoughi, Roy & Aral (2018). The spread of true and false news online. *Science*.
- Thorne et al. (2018). FEVER: a large-scale dataset for Fact Extraction and
  VERification. *NAACL-HLT*.
- Nielsen & McConville (2022). MuMiN: A Large-Scale Multilingual Multimodal
  Fact-Checked Misinformation Social Network Dataset. *Findings of ACL*.
