"""
Statistical comparison utilities for PHEME spread analysis.

Three kinds of tests, each with a clear purpose for the presentation:

  1) Per-event Mann-Whitney U + Cliff's delta:
     For each event with adequate sample sizes, test pairwise veracity
     comparisons. Report effect size (Cliff's delta) AND p-value, because
     with ~6,400 threads p-values become misleadingly tiny.

  2) Pooled stratified test:
     Combine evidence across events while controlling for event-level
     differences. We use a stratified Mann-Whitney via van Elteren's test,
     which is the right pooled test when groups are unbalanced across
     strata (which they are here — e.g. ferguson is mostly nonrumour,
     prince-toronto is mostly false).

  3) Consistency score:
     For each metric and pairwise comparison, count: "in how many of N
     eligible events does class A > class B?" This is the most
     presentable finding format — it's robust, easy to state, and
     resistant to single-event outliers.

  4) Bootstrap confidence intervals for per-event medians:
     For Phase 4's per-event figure. Resampling-based CIs, not parametric.

WHY THESE CHOICES

Mann-Whitney U is the standard non-parametric two-sample test. It's
appropriate here because cascade metrics are heavy-tailed and not
remotely normal. We do NOT use t-tests.

Cliff's delta is the rank-biserial correlation, ranging from -1 to +1:
    delta =  P(X > Y) - P(X < Y)
where X is drawn from group A and Y from group B. Romano et al. (2006)
suggest interpretation thresholds:
    |delta| < 0.147  → negligible
    0.147 ≤ |delta| < 0.33  → small
    0.33 ≤ |delta| < 0.474  → medium
    |delta| ≥ 0.474  → large
A delta of +0.2 means: a random A-thread is ~20 percentage points more
likely to exceed a random B-thread than the reverse.

Bootstrap CIs at 95% use the percentile method (simple, defensible, and
robust to skewness). 1000 resamples is standard for medians.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cliff's delta: P(A > B) - P(A < B). Returns NaN if either group empty.

    Implementation: compute via Mann-Whitney U statistic to avoid the
    O(n*m) pairwise comparison. delta = 2U / (n*m) - 1.
    """
    a = np.asarray([x for x in a if not pd.isna(x)])
    b = np.asarray([x for x in b if not pd.isna(x)])
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return np.nan
    # scipy's U is sum-of-ranks-in-A minus n_a*(n_a+1)/2; equivalently,
    # the count of (a_i, b_j) pairs with a_i > b_j (with 0.5 for ties).
    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(2 * u / (n_a * n_b) - 1)


def cliffs_delta_magnitude(delta: float) -> str:
    """Romano et al. (2006) interpretation thresholds."""
    if pd.isna(delta):
        return "n/a"
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Per-event pairwise tests
# ---------------------------------------------------------------------------

def per_event_pairwise(
    df: pd.DataFrame,
    metric: str,
    veracity_pairs: list[tuple[str, str]],
    min_n_per_group: int = 30,
) -> pd.DataFrame:
    """
    For every (event, veracity_pair) combination, run Mann-Whitney U +
    Cliff's delta. Skip combinations where either group has fewer than
    `min_n_per_group` non-null observations.

    Returns a long dataframe with columns:
        event, metric, group_a, group_b, n_a, n_b, median_a, median_b,
        delta, magnitude, p_value, eligible

    Why min_n=30? It's the rule of thumb where Mann-Whitney's normal
    approximation becomes reasonable, and below ~30 effect-size estimates
    are unstable. Tunable — pass min_n_per_group=10 for an exploratory
    pass that includes more events.
    """
    rows = []
    for event in sorted(df["event"].unique()):
        sub = df[df["event"] == event]
        for a, b in veracity_pairs:
            xs = sub[sub["veracity"] == a][metric].dropna()
            ys = sub[sub["veracity"] == b][metric].dropna()
            n_a, n_b = len(xs), len(ys)
            eligible = (n_a >= min_n_per_group) and (n_b >= min_n_per_group)
            if eligible:
                delta = cliffs_delta(xs, ys)
                _, p = stats.mannwhitneyu(xs, ys, alternative="two-sided")
            else:
                delta = np.nan
                p = np.nan
            rows.append({
                "event": event,
                "metric": metric,
                "group_a": a,
                "group_b": b,
                "n_a": n_a,
                "n_b": n_b,
                "median_a": float(xs.median()) if n_a else np.nan,
                "median_b": float(ys.median()) if n_b else np.nan,
                "delta": delta,
                "magnitude": cliffs_delta_magnitude(delta),
                "p_value": p,
                "eligible": eligible,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pooled stratified test (van Elteren)
# ---------------------------------------------------------------------------

def van_elteren_test(
    df: pd.DataFrame,
    metric: str,
    group_a: str,
    group_b: str,
    stratum_col: str = "event",
    min_n_per_group: int = 30,
) -> dict:
    """
    Stratified Mann-Whitney test (van Elteren 1960) — the appropriate
    pooled test when groups are unbalanced across strata.

    For each eligible stratum (event), compute Mann-Whitney U and combine
    via inverse-variance weighting. Returns a dict with combined Z, p,
    and a weighted average Cliff's delta (effect size across strata).

    Skips strata where either group has <min_n_per_group observations.
    Returns NaN if no strata are eligible.

    REFERENCE
    van Elteren, P. H. (1960). On the combination of independent two-sample
    tests of Wilcoxon. Bulletin of the International Statistical Institute.
    """
    eligible_strata = []
    for stratum in df[stratum_col].unique():
        sub = df[df[stratum_col] == stratum]
        xs = sub[sub["veracity"] == group_a][metric].dropna()
        ys = sub[sub["veracity"] == group_b][metric].dropna()
        if len(xs) < min_n_per_group or len(ys) < min_n_per_group:
            continue
        eligible_strata.append((stratum, xs, ys))

    if not eligible_strata:
        return {
            "combined_z": np.nan, "combined_p": np.nan,
            "weighted_delta": np.nan, "n_strata": 0,
            "strata_used": [],
        }

    # van Elteren weighting: each stratum contributes Z_h with weight
    # w_h = 1 / (n_a + n_b + 1). Combined Z = Σ(w_h * Z_h) / sqrt(Σ w_h²).
    z_terms, weights, deltas, weights_for_delta = [], [], [], []
    for stratum, xs, ys in eligible_strata:
        n_a, n_b = len(xs), len(ys)
        u, _ = stats.mannwhitneyu(xs, ys, alternative="two-sided")
        # Standardize U → Z (continuity correction omitted; negligible
        # at these sample sizes)
        mean_u = n_a * n_b / 2
        var_u = n_a * n_b * (n_a + n_b + 1) / 12
        z = (u - mean_u) / np.sqrt(var_u)
        w = 1 / (n_a + n_b + 1)
        z_terms.append(w * z)
        weights.append(w**2)
        deltas.append(2 * u / (n_a * n_b) - 1)
        weights_for_delta.append(n_a * n_b)  # Mantel-Haenszel-style weighting

    combined_z = sum(z_terms) / np.sqrt(sum(weights))
    combined_p = 2 * (1 - stats.norm.cdf(abs(combined_z)))
    weighted_delta = np.average(deltas, weights=weights_for_delta)

    return {
        "combined_z": float(combined_z),
        "combined_p": float(combined_p),
        "weighted_delta": float(weighted_delta),
        "n_strata": len(eligible_strata),
        "strata_used": [s[0] for s in eligible_strata],
    }


# ---------------------------------------------------------------------------
# Multiple testing correction
# ---------------------------------------------------------------------------

def bh_correction(
    per_event_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction to eligible rows of a
    per_event_pairwise result dataframe. Adds two columns:
      p_value_adj:    BH-adjusted p-value (NaN for ineligible rows)
      significant_bh: bool, whether p_value_adj < alpha

    WHY THIS MATTERS
    Running per_event_pairwise over 11 metrics × 5 pairs × 9 events
    produces up to 495 simultaneous tests. At α=0.05, roughly 25 would
    be false positives by chance alone. BH controls the *expected
    proportion* of false discoveries (the false discovery rate) rather
    than the family-wise error rate — appropriate here because we are
    doing exploratory analysis, not confirming a single hypothesis.

    Use this as a final filter before constructing the consistency table:
        results_corrected = bh_correction(stats_per_event)
        consistency = consistency_score(results_corrected, use_adjusted_p=True)
    """
    df = per_event_df.copy()
    eligible_mask = df["eligible"] & df["p_value"].notna()
    df["p_value_adj"] = np.nan
    df["significant_bh"] = False

    if eligible_mask.sum() == 0:
        return df

    p_vals = df.loc[eligible_mask, "p_value"].values
    n = len(p_vals)
    order = np.argsort(p_vals)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)

    # BH step-up: adjusted p = min(p_i * n / rank_i, 1.0)
    # Enforce monotonicity by scanning from largest to smallest rank.
    adj = np.minimum(p_vals * n / ranks, 1.0)
    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])

    df.loc[eligible_mask, "p_value_adj"] = adj
    df["significant_bh"] = df["p_value_adj"] < alpha
    return df


# ---------------------------------------------------------------------------
# Consistency scoring
# ---------------------------------------------------------------------------

def consistency_score(
    per_event_df: pd.DataFrame,
    direction: str = "a_greater",
    alpha: float = 0.05,
    use_adjusted_p: bool = False,
) -> pd.DataFrame:
    """
    Given the output of per_event_pairwise, summarize per (metric, pair):
      n_eligible:          how many events had enough data
      n_a_greater:         events where median_a > median_b AND p < alpha
      n_b_greater:         events where median_b > median_a AND p < alpha
      n_no_diff:           events where p >= alpha
      consistency_a_gt_b:  n_a_greater / n_eligible
      consistency_b_gt_a:  n_b_greater / n_eligible

    `direction='a_greater'` ranks results by how often A > B (the typical
    presentation framing). Pass 'b_greater' to flip.

    use_adjusted_p:
      If True, use the `p_value_adj` column (BH-corrected) instead of
      raw p-values. Requires that bh_correction() has been called on
      per_event_df first. Recommended for the final presentation table.

    Why this matters for the presentation: a finding of "delta = 0.15,
    p < 0.001" sounds strong but is hard to defend. A finding of
    "unverified > nonrumour in 5 of 5 eligible events, all with p<0.01,
    median delta = 0.18" is much harder to argue with. Consistency is
    the killer slide format.
    """
    p_col = (
        "p_value_adj"
        if use_adjusted_p and "p_value_adj" in per_event_df.columns
        else "p_value"
    )
    rows = []
    for (metric, a, b), grp in per_event_df.groupby(["metric", "group_a", "group_b"]):
        elig = grp[grp["eligible"]]
        n_eligible = len(elig)
        if n_eligible == 0:
            rows.append({
                "metric": metric, "group_a": a, "group_b": b,
                "n_eligible": 0, "n_a_greater": 0, "n_b_greater": 0,
                "n_no_diff": 0, "consistency_a_gt_b": np.nan,
                "consistency_b_gt_a": np.nan, "median_delta": np.nan,
            })
            continue
        sig = elig[elig[p_col] < alpha]
        n_a_greater = ((sig["median_a"] > sig["median_b"])).sum()
        n_b_greater = ((sig["median_b"] > sig["median_a"])).sum()
        rows.append({
            "metric": metric,
            "group_a": a,
            "group_b": b,
            "n_eligible": n_eligible,
            "n_a_greater": int(n_a_greater),
            "n_b_greater": int(n_b_greater),
            "n_no_diff": n_eligible - int(n_a_greater) - int(n_b_greater),
            "consistency_a_gt_b": n_a_greater / n_eligible,
            "consistency_b_gt_a": n_b_greater / n_eligible,
            "median_delta": float(elig["delta"].median()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap CI for per-event medians
# ---------------------------------------------------------------------------

def bootstrap_median_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Returns (median, ci_low, ci_high) using percentile bootstrap.
    Returns (NaN, NaN, NaN) if values is empty after dropping NaN.
    """
    arr = np.asarray([v for v in values if not pd.isna(v)])
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    medians = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        medians[i] = np.median(sample)
    return (
        float(np.median(arr)),
        float(np.quantile(medians, alpha / 2)),
        float(np.quantile(medians, 1 - alpha / 2)),
    )


def per_event_medians_with_ci(
    df: pd.DataFrame,
    metric: str,
    veracity_classes: list[str],
    n_boot: int = 1000,
) -> pd.DataFrame:
    """
    Compute median + 95% bootstrap CI per (event, veracity) for one metric.
    Returns long-form dataframe ready for plotting.
    """
    rows = []
    for event in sorted(df["event"].unique()):
        for v in veracity_classes:
            vals = df[(df["event"] == event) & (df["veracity"] == v)][metric].dropna()
            med, lo, hi = bootstrap_median_ci(vals, n_boot=n_boot)
            rows.append({
                "event": event, "veracity": v,
                "n": len(vals),
                "median": med, "ci_low": lo, "ci_high": hi,
            })
    return pd.DataFrame(rows)
