"""
Phase 4: presentation figures.

Six figures, each carrying a specific piece of narrative work:

  fig1_hook_cascade_trees:    side-by-side cascade trees (rumour vs nonrumour)
  fig2_speed_cdfs:            CDFs of time-to-first-reply by veracity
  fig3_velocity_consistency:  per-event consistency of speed effects
  fig4_reach_null:            cascade size by veracity (the null finding)
  fig5_structure_null:        depth vs size scatter, no veracity clustering
  fig6_methodology:           pooled vs stratified comparison

Design principles (these matter for a presentation):
  - One figure = one idea. Don't pack two findings into one chart.
  - Consistent veracity colors across all figures (set in VERACITY_COLORS).
  - Log scale on heavy-tailed metrics (cascade size, time, velocity).
  - Honest uncertainty: bootstrap CIs, not just point estimates.
  - Sans-serif body, but readable on a projector — minimum 11pt for axis text.
  - Save as both PNG (for slides) and PDF (for reports).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


VERACITY_ORDER = ["nonrumour", "true", "unverified", "false"]
VERACITY_COLORS = {
    "nonrumour":  "#4C72B0",
    "true":       "#55A868",
    "unverified": "#DD8452",
    "false":      "#C44E52",
}
VERACITY_LABELS = {
    "nonrumour":  "Non-rumour",
    "true":       "True rumour",
    "unverified": "Unverified",
    "false":      "False rumour",
}


def _setup_style():
    """Consistent matplotlib styling. Call once at notebook start."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, path_base: Path):
    """Save as both PNG (slides) and PDF (reports)."""
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_base}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path_base}.pdf", bbox_inches="tight", facecolor="white")


# ---------------------------------------------------------------------------
# Figure 1 — Hook: side-by-side cascade trees
# ---------------------------------------------------------------------------

def fig1_hook_cascade_trees(
    edges_df: pd.DataFrame,
    threads_df: pd.DataFrame,
    rumour_thread_id: str,
    nonrumour_thread_id: str,
    out_path: Path,
):
    """
    Side-by-side reply trees: one rumour, one non-rumour. Pick threads
    from the SAME EVENT for fair comparison. Aim for similar cascade
    sizes if possible — the point is to show structural difference, not
    size difference.

    The radial "circo" layout from graphviz isn't always available, so
    we use NetworkX's spring layout with a depth-based vertical hierarchy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, thread_id, label, color in [
        (axes[0], rumour_thread_id, "Rumour", VERACITY_COLORS["unverified"]),
        (axes[1], nonrumour_thread_id, "Non-rumour", VERACITY_COLORS["nonrumour"]),
    ]:
        sub = edges_df[edges_df["thread_id"] == thread_id]
        g = nx.DiGraph()
        g.add_node(thread_id)
        for _, e in sub.iterrows():
            g.add_edge(e["parent_id"], e["child_id"])

        # Hierarchical layout: depth determines y, sibling order determines x
        depths = nx.single_source_shortest_path_length(g, thread_id)
        max_depth = max(depths.values()) if depths else 0
        # Group nodes by depth, then space them evenly
        pos = {}
        by_depth = {}
        for n, d in depths.items():
            by_depth.setdefault(d, []).append(n)
        for d, nodes in by_depth.items():
            for i, n in enumerate(nodes):
                x = (i + 0.5) / len(nodes) - 0.5
                pos[n] = (x, -d)

        # Color: source is dark, replies are the veracity color
        node_colors = [
            "#222222" if n == thread_id else color
            for n in g.nodes()
        ]
        node_sizes = [
            120 if n == thread_id else 30
            for n in g.nodes()
        ]
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#aaaaaa", width=0.5,
                               arrows=False, alpha=0.7)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.9, linewidths=0)

        n_nodes = g.number_of_nodes()
        ax.set_title(f"{label}: {n_nodes} tweets, depth {max_depth}",
                     fontsize=12, fontweight="bold", color=color)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)

    fig.suptitle("How conversations form around breaking news",
                 fontsize=14, fontweight="bold", y=1.02)
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Speed CDFs
# ---------------------------------------------------------------------------

def fig2_speed_cdfs(
    tm: pd.DataFrame,
    metric: str = "time_to_first_reply_min",
    out_path: Path | None = None,
    title: str | None = None,
    xlabel: str | None = None,
):
    """
    Empirical CDF of `metric` for each veracity class. CDFs are the
    honest way to show distributional differences — no binning choices,
    no aggregation, all data points visible.

    A CDF curve to the LEFT means values are smaller (faster, in time
    metrics). The visual question for the audience: do the curves
    separate, and which is leftmost?
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for v in VERACITY_ORDER:
        vals = tm[tm["veracity"] == v][metric].dropna()
        vals = vals[vals > 0]  # drop zeros for log scale; honest because log(0) is undef
        if len(vals) == 0:
            continue
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf,
                color=VERACITY_COLORS[v], lw=2,
                label=f"{VERACITY_LABELS[v]} (n={len(vals):,})")

    ax.set_xscale("log")
    ax.set_xlabel(xlabel or metric.replace("_", " ").title())
    ax.set_ylabel("Cumulative fraction of cascades")
    ax.set_title(title or "Distribution of cascade speed by veracity")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 1.02)

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Per-event consistency
# ---------------------------------------------------------------------------

def fig3_velocity_consistency(
    consistency_df: pd.DataFrame,
    pair: tuple[str, str] = ("unverified", "nonrumour"),
    metrics_to_show: list | None = None,
    out_path: Path | None = None,
):
    """
    Stacked horizontal bars: for each metric, how many of the eligible
    events showed (a > b), (b > a), or (no significant difference)?

    Good for the "consistency" story slide: a quick visual answer to
    "is this finding robust across events or driven by one event?"
    """
    metrics_to_show = metrics_to_show or [
        "time_to_first_reply_min", "time_to_half_cascade_min",
        "reply_velocity_first_hour", "cascade_size", "max_depth",
        "structural_virality", "broadcast_ratio",
    ]

    a, b = pair
    sub = consistency_df[
        (consistency_df["group_a"] == a) &
        (consistency_df["group_b"] == b) &
        (consistency_df["metric"].isin(metrics_to_show)) &
        (consistency_df["n_eligible"] >= 3)
    ].copy()

    # Order metrics by strength of finding (consistency in either direction)
    sub["max_consistency"] = sub[
        ["consistency_a_gt_b", "consistency_b_gt_a"]
    ].max(axis=1)
    sub = sub.sort_values("max_consistency", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(sub))))
    y = np.arange(len(sub))

    ax.barh(y, sub["n_a_greater"], color=VERACITY_COLORS[a],
            label=f"{VERACITY_LABELS[a]} > {VERACITY_LABELS[b]}", height=0.7)
    ax.barh(y, sub["n_b_greater"], left=sub["n_a_greater"],
            color=VERACITY_COLORS[b],
            label=f"{VERACITY_LABELS[b]} > {VERACITY_LABELS[a]}", height=0.7)
    ax.barh(y, sub["n_no_diff"],
            left=sub["n_a_greater"] + sub["n_b_greater"],
            color="#dddddd", label="No significant difference", height=0.7)

    # Pretty metric labels
    pretty = {
        "cascade_size": "Cascade size",
        "max_depth": "Tree depth",
        "max_breadth": "Tree breadth",
        "unique_users": "Unique users",
        "time_to_first_reply_min": "Time to first reply",
        "time_to_half_cascade_min": "Time to half-cascade",
        "reply_velocity_first_hour": "Replies in first hour",
        "structural_virality": "Structural virality",
        "wiener_index": "Wiener index",
        "broadcast_ratio": "Broadcast ratio",
        "branching_factor_mean": "Branching factor",
    }
    ax.set_yticks(y)
    ax.set_yticklabels([pretty.get(m, m) for m in sub["metric"]])
    ax.set_xlabel(f"Number of events (out of {sub['n_eligible'].max()} eligible)")
    ax.set_title(
        f"Where the {VERACITY_LABELS[a]} vs {VERACITY_LABELS[b]} pattern holds\n"
        f"(per-event Mann-Whitney p<0.05)"
    )
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(False, axis="y")

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Reach is null
# ---------------------------------------------------------------------------

def fig4_reach_null(
    tm: pd.DataFrame,
    metric: str = "cascade_size",
    out_path: Path | None = None,
):
    """
    Boxplot of `metric` by veracity. The story: there's no clean
    veracity ordering on this metric — supports the null finding for
    reach. Log scale because the distributions are heavy-tailed.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    data = tm[tm[metric].notna() & (tm[metric] > 0)]
    sns.boxplot(
        data=data, x="veracity", y=metric,
        order=VERACITY_ORDER, hue="veracity",
        palette=[VERACITY_COLORS[v] for v in VERACITY_ORDER],
        showfliers=False, ax=ax, width=0.6, linewidth=1.2,
        legend=False,
    )
    ax.set_yscale("log")
    pretty = {
        "cascade_size": "Cascade size (tweets)",
        "max_depth": "Maximum reply depth",
        "unique_users": "Unique users in cascade",
    }
    ax.set_ylabel(pretty.get(metric, metric))
    ax.set_xlabel("")
    ax.set_xticks(range(len(VERACITY_ORDER)))
    ax.set_xticklabels([VERACITY_LABELS[v] for v in VERACITY_ORDER])
    ax.set_title(f"{pretty.get(metric, metric)}: little difference across veracity")

    # Annotate sample sizes
    for i, v in enumerate(VERACITY_ORDER):
        n = (data["veracity"] == v).sum()
        ax.annotate(f"n={n:,}", xy=(i, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, -25), textcoords="offset points",
                    ha="center", fontsize=9, color="#555555")

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Structure is null
# ---------------------------------------------------------------------------

def fig5_structure_null(
    tm: pd.DataFrame,
    out_path: Path | None = None,
):
    """
    Depth vs cascade size, colored by veracity. If veracity classes
    cluster differently, you'd see color separation. They don't, so
    the "rumours have a different structural shape" story doesn't hold.

    This is the figure that PROVES the null structure finding visually.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    data = tm[
        tm["cascade_size"].notna() & (tm["cascade_size"] > 1) &
        tm["max_depth"].notna() & (tm["max_depth"] > 0)
    ].copy()

    # Plot non-rumours first (largest group, dimmer), rumours on top
    for v in ["nonrumour", "true", "unverified", "false"]:
        sub = data[data["veracity"] == v]
        ax.scatter(sub["cascade_size"], sub["max_depth"],
                   c=VERACITY_COLORS[v], alpha=0.35,
                   s=15, edgecolors="none",
                   label=f"{VERACITY_LABELS[v]} (n={len(sub):,})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cascade size (tweets)")
    ax.set_ylabel("Maximum reply depth")
    ax.set_title("No veracity clustering in cascade shape\n"
                 "(if rumours had a distinct structural shape, colors would separate)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.3)

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Methodology: pooled vs. stratified
# ---------------------------------------------------------------------------

def fig_per_event_medians_ci(
    medians_df: pd.DataFrame,
    metric: str,
    veracity_classes: list | None = None,
    min_n: int = 10,
    out_path: Path | None = None,
):
    """
    Per-event panel: median ± 95% bootstrap CI for each veracity class.

    medians_df is the output of per_event_medians_with_ci() from
    stats_comparison.py. Each event gets its own subplot; within each
    subplot, one dot + error bar per veracity class with enough data.

    This figure is the companion to fig3_velocity_consistency — it shows
    the *magnitude* of the per-event differences, not just the direction.
    Use it as a supplementary slide or in the report.

    min_n: minimum observations per (event, veracity) cell to plot.
    """
    veracity_classes = veracity_classes or VERACITY_ORDER
    sub = medians_df[
        (medians_df["metric"] == metric) &
        (medians_df["n"] >= min_n) &
        (medians_df["veracity"].isin(veracity_classes))
    ].copy()

    events = sorted(sub["event"].unique())
    if not events:
        raise ValueError(f"No events with n >= {min_n} for metric='{metric}'")

    ncols = min(3, len(events))
    nrows = (len(events) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharey=False,
    )
    axes_flat = np.array(axes).flatten() if len(events) > 1 else [axes]

    for ax, event in zip(axes_flat, events):
        ev = sub[sub["event"] == event]
        for i, v in enumerate(veracity_classes):
            row = ev[ev["veracity"] == v]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            ax.errorbar(
                x=i,
                y=r["median"],
                yerr=[[r["median"] - r["ci_low"]], [r["ci_high"] - r["median"]]],
                fmt="o",
                color=VERACITY_COLORS[v],
                capsize=5,
                ms=8,
                linewidth=1.5,
                label=VERACITY_LABELS[v],
            )

        ax.set_title(event, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(veracity_classes)))
        ax.set_xticklabels(
            [VERACITY_LABELS[v] for v in veracity_classes],
            rotation=25, ha="right", fontsize=8,
        )
        ax.set_ylabel(metric.replace("_", " "), fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes_flat[len(events):]:
        ax.axis("off")

    fig.suptitle(
        f"{metric.replace('_', ' ').title()} — median ± 95% CI by event",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if out_path:
        _save(fig, out_path)
    return fig


def fig6_methodology_flip(
    tm: pd.DataFrame,
    out_path: Path | None = None,
    metric: str = "cascade_size",
):
    """
    Two-panel figure showing how the apparent finding flipped between
    pooled and per-event analysis. LEFT: pooled bar chart suggesting
    non-rumours are biggest. RIGHT: per-event panel showing the ranking
    actually varies by event.

    This is the figure for the methodology slide. Its job is to make
    the audience FEEL the difference between the two analyses.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1, 2]})

    # LEFT: pooled medians
    pooled = tm.groupby("veracity")[metric].median().reindex(VERACITY_ORDER)
    bars = axes[0].bar(
        range(len(pooled)), pooled.values,
        color=[VERACITY_COLORS[v] for v in pooled.index],
        edgecolor="white", linewidth=1.5,
    )
    axes[0].set_xticks(range(len(pooled)))
    axes[0].set_xticklabels([VERACITY_LABELS[v] for v in pooled.index],
                            rotation=20, ha="right")
    axes[0].set_ylabel(f"Median {metric.replace('_', ' ')}")
    axes[0].set_title('Pooled view\n"Non-rumours have biggest cascades"')
    for bar, v in zip(bars, pooled.values):
        axes[0].annotate(f"{v:.0f}", xy=(bar.get_x() + bar.get_width()/2, v),
                         ha="center", va="bottom", fontsize=10)

    # RIGHT: per-event medians, only events with all 4 classes meaningful
    eligible_events = [
        e for e in sorted(tm["event"].unique())
        if (tm[tm["event"] == e]["veracity"].value_counts() >= 30).sum() >= 3
    ]
    sub = tm[tm["event"].isin(eligible_events)]
    per_event = (sub.groupby(["event", "veracity"])[metric]
                 .median().unstack("veracity")
                 .reindex(columns=VERACITY_ORDER))

    x = np.arange(len(eligible_events))
    width = 0.2
    for i, v in enumerate(VERACITY_ORDER):
        if v in per_event.columns:
            vals = per_event[v].values
            axes[1].bar(x + i * width - 1.5 * width, vals,
                        width=width, color=VERACITY_COLORS[v],
                        label=VERACITY_LABELS[v], edgecolor="white", linewidth=0.5)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(eligible_events, rotation=25, ha="right")
    axes[1].set_ylabel(f"Median {metric.replace('_', ' ')}")
    axes[1].set_title("Per-event view\n"
                      "Ranking varies by event — pooled finding is event-confounded")
    axes[1].legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.95)

    fig.suptitle("Why analysis choices matter: pooled vs. stratified comparisons",
                 fontsize=13, fontweight="bold", y=1.02)

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 7 — Verifiability score distribution by veracity
# ---------------------------------------------------------------------------

def fig_verifiability_by_veracity(
    tm,
    score_col: str = "verifiability_score",
    out_path=None,
):
    """
    Overlapping histograms of P(VERIFIABLE) by veracity class.

    The story: unverified and false rumours cluster toward lower verifiability
    scores, supporting the mechanism that hard-to-check claims spread faster
    because readers cannot easily debunk them before sharing.

    Cite: Nielsen & McConville (2022) MuMiN; Thorne et al. (2018) FEVER.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for v in VERACITY_ORDER:
        vals = tm[tm["veracity"] == v][score_col].dropna()
        if len(vals) == 0:
            continue
        ax.hist(
            vals, bins=40, alpha=0.55, density=True,
            color=VERACITY_COLORS[v],
            label=f"{VERACITY_LABELS[v]} (n={len(vals):,})",
        )

    ax.axvline(0.5, color="#888888", linewidth=1, linestyle=":", label="decision boundary")
    ax.set_xlabel("P(verifiable) — FEVER-trained classifier", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Unverified rumours are linguistically less verifiable\n"
        "(classifier trained on FEVER, Thorne et al. 2018)",
        fontweight="bold",
    )
    ax.legend(frameon=True, framealpha=0.9, fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 8 — Verifiability quartile vs spread speed (mechanism slide)
# ---------------------------------------------------------------------------

def fig_verifiability_speed_quartiles(
    tm,
    speed_col: str = "time_to_first_reply_min",
    score_col: str = "verifiability_score",
    out_path=None,
):
    """
    Median time-to-first-reply by verifiability quartile.

    The mechanism slide: Q1 (least verifiable) → fastest replies,
    Q4 (most verifiable) → slowest. A monotonic pattern here is the
    cleanest possible evidence for the verifiability-as-mechanism hypothesis.

    Error bars show IQR (25th–75th percentile), not SE — honest for
    heavy-tailed distributions.
    """
    valid = tm[tm[speed_col].notna() & (tm[speed_col] > 0) & tm[score_col].notna()].copy()
    valid["verif_quartile"] = pd.qcut(
        valid[score_col], q=4,
        labels=["Q1\n(least\nverifiable)", "Q2", "Q3", "Q4\n(most\nverifiable)"],
    )
    grp = (
        valid.groupby("verif_quartile", observed=True)[speed_col]
        .agg(median="median", n="count",
             p25=lambda x: x.quantile(0.25),
             p75=lambda x: x.quantile(0.75))
        .reset_index()
    )

    from scipy import stats as _stats
    rho, p_rho = _stats.spearmanr(valid[score_col], np.log(valid[speed_col]))

    palette = ["#C44E52", "#DD8452", "#55A868", "#4C72B0"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (_, row) in enumerate(grp.iterrows()):
        ax.bar(i, row["median"], color=palette[i],
               edgecolor="white", linewidth=1.2, width=0.65)
        ax.errorbar(
            i, row["median"],
            yerr=[[row["median"] - row["p25"]], [row["p75"] - row["median"]]],
            fmt="none", color="#222222", capsize=6, linewidth=1.8,
        )
        ax.text(i, row["p75"] + 0.5, f"n={int(row['n']):,}",
                ha="center", va="bottom", fontsize=9, color="#555555")

    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(grp["verif_quartile"].astype(str), fontsize=10)
    ax.set_ylabel("Median time to first reply (minutes)", fontsize=11)
    ax.set_title(
        f"Less verifiable tweets attract faster replies\n"
        f"Spearman ρ = {rho:.2f}, p = {p_rho:.1e}  |  bars = median, whiskers = IQR",
        fontweight="bold",
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if out_path:
        _save(fig, out_path)
    return fig
