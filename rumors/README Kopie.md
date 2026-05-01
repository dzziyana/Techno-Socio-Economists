# PHEME Spread Dynamics — Group Project

Comparative spread analysis of rumours vs. facts during breaking-news
events, using the PHEME-9 dataset.

## Central thesis (after Phase 3 results)

> **During breaking-news events, rumours generate faster early engagement than non-rumours, but reach comparable size and structural shape.**

The famous "falsehoods spread farther, faster, deeper" claim from
Vosoughi et al. (Science, 2018) only partially replicates on PHEME's
breaking-news events. The speed dimension survives careful analysis;
the reach and shape dimensions don't.

## Project layout

```
pheme_project/
├── data/
│   ├── all-rnr-annotated-threads/   ← PUT THE REAL PHEME DATA HERE
│   ├── sample/                       ← small toy dataset for testing
│   └── processed/                    ← generated dataframes
├── src/
│   ├── pheme_loader.py               ← Phase 1: parsing logic
│   ├── cascade_metrics.py            ← Phase 2: 11 metrics per thread
│   ├── stats_comparison.py           ← Phase 3: tests, effect sizes, CIs
│   └── figures.py                    ← Phase 4: 6 presentation figures
├── notebooks/
│   ├── 01_data_wrangling.ipynb        ← Phase 1: load & validate
│   ├── 02_cascade_metrics.ipynb       ← Phase 2: compute metrics
│   ├── 03_statistical_comparison.ipynb ← Phase 3: rigor
│   └── 04_presentation_figures.ipynb  ← Phase 4: figures + storyboard
└── figures/                           ← saved figures (PNG + PDF)
```

## How to run end-to-end

1. Download PHEME from Figshare ("PHEME dataset for Rumour Detection
   and Veracity Classification", Zubiaga et al.).

2. Install dependencies: `pip install pandas pyarrow networkx matplotlib seaborn scipy`

3. Run the four notebooks in order (1 → 2 → 3 → 4). Total runtime
   end-to-end: ~2-3 minutes.

4. After running 04, the `figures/` directory contains 6 PNG/PDF pairs
   ready to drop into a slide deck.

## Data quality issues fixed during the project

Four real issues with PHEME's distribution surfaced and were patched.
Worth one methodology slide:

1. **Curly quotes in `structure.json`** for some ferguson/ottawashooting
   threads. JSON parser accepts them but collapses keys. Fix: sanitize
   Unicode quotes before parsing.
2. **Non-numeric placeholder keys** representing unresolvable tweet
   IDs at collection time. Fix: skip these nodes and descendants.
3. **Source tweet duplicated in `reactions/`** for ~88% of ferguson
   threads — inflated cascade sizes and zeroed first-reply times.
   Fix: skip reaction files matching the thread ID.
4. **Heterogeneous veracity annotation schemas.** Veracity is encoded
   in `true` and `misinformation` flags, not in `category` (which is
   the rumour description). Fix: normalize via flags only.

## Key methodology decisions (defensible under questions)

- **Trust `structure.json` for tree topology**, not `in_reply_to_status_id`.
- **`n_tweets_in_structure` is cascade size**, not `n_reactions`.
- **Four veracity classes**: `true`, `false`, `unverified`, `nonrumour`.
- **Per-event stratification, not pooled** — pooled comparisons on
  PHEME are dominated by event-level differences.
- **Effect sizes (Cliff's delta), not just p-values.** Romano et al.
  (2006) thresholds: |d|<0.147 negligible, <0.33 small, <0.474 medium,
  ≥0.474 large.

## Group division (for 4 people)

1. **Data engineering** (Phase 1)
2. **Network metrics** (Phase 2)
3. **Statistical analysis** (Phase 3)
4. **Narrative & visualization** (Phase 4)

## Limitations

- PHEME is small (~6,400 threads, 9 events) and English-only.
- Reply trees, not full retweet networks.
- Snapshot in time — cascades may have continued after collection.
- Source tweets for non-rumours were chosen from authoritative news
  accounts, which inflates non-rumour reach via follower-count effects.
