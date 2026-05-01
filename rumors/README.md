# PHEME Spread Dynamics — Group Project

Comparative spread analysis of rumors vs. facts during breaking news events,
using the PHEME-9 dataset.

## Story arc

> **Do rumors really spread faster and farther than the truth — and what shape does that spread take during breaking news?**

Vosoughi et al. (*Science*, 2018) found that falsehoods spread "farther,
faster, deeper, and more broadly" than truth on Twitter. Our project asks
whether this still holds during breaking-news events specifically — when
facts haven't been established yet and unverified claims fill the vacuum.

## Project layout

```
pheme_project/
├── data/
│   ├── sample/              ← small toy dataset for testing the pipeline
│   ├── all-rnr-annotated-threads/   ← PUT THE REAL PHEME DATA HERE
│   └── processed/           ← cleaned dataframes (output of Phase 1)
├── src/
│   └── pheme_loader.py      ← parsing logic, importable as a module
├── notebooks/
│   └── 01_data_wrangling.ipynb   ← Phase 1: load, validate, save
└── figures/                 ← presentation figures (Phase 4 output)
```

## How to run Phase 1

1. Download PHEME from Figshare
   (`PHEME dataset for Rumour Detection and Veracity Classification`,
   Zubiaga et al.). Unzip so the path
   `data/all-rnr-annotated-threads/charliehebdo-all-rnr-threads/...` exists.
2. Install dependencies: `pip install pandas pyarrow networkx matplotlib seaborn scipy`
3. Open `notebooks/01_data_wrangling.ipynb`, change the `PHEME_ROOT` path
   in cell 2 to point at the real data, and run all cells.
4. Confirm the outputs in `data/processed/` look reasonable.

Expected output on full PHEME-9: ~6,400 threads, ~100k tweets, ~95k edges.

## Phase plan

- **Phase 1 (this notebook):** parse PHEME → tidy dataframes. ✓
- **Phase 2:** compute cascade metrics (speed, reach, structural virality).
- **Phase 3:** statistical comparison across veracity classes.
- **Phase 4:** the six presentation figures.

## Group division (suggestion for 4 people)

1. **Data engineering** — own the loader, validate completeness, handle
   weird events / corrupt files.
2. **Network metrics** — NetworkX cascade trees, depth/breadth/structural
   virality. (Phase 2)
3. **Statistical analysis** — Mann-Whitney tests, per-event stratification,
   the "is the difference real" question. (Phase 3)
4. **Narrative & visualization** — the story arc, slide design, the six
   figures. Should start sketching the deck in week 1, not week 3.

## Key design decisions (so you can defend them)

- **We trust `structure.json`, not `in_reply_to_status_id`.** The structure
  file is the canonical reply tree; some replies in the tree have deleted
  JSONs but the tree topology is preserved.
- **`n_tweets_in_structure` is our cascade size**, not `n_reactions`.
  Reactions counts what we have JSONs for; the tree counts what actually
  spread.
- **Veracity has four classes**: `true`, `false`, `unverified`, `nonrumour`.
  Don't collapse `unverified` into either rumour bucket — it's a distinct
  category with its own dynamics, and may turn out to be the most
  interesting one.
- **Always report stratified by event.** Some events are dominated by one
  or two viral cascades; pooled stats can be misleading.

## Limitations to acknowledge in the presentation

- PHEME is small (~6,400 threads, 9 events) and English-only.
- Reply trees, not full retweet networks — depth is conversational, not
  diffusional.
- Snapshot in time — cascades may have continued after collection.
- Annotations are journalist-curated, not crowdsourced — high precision
  but limited recall.
