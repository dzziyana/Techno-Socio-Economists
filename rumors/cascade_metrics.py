"""
Cascade metrics for PHEME spread analysis.

Given the three dataframes from Phase 1 (threads, tweets, edges), compute
a per-thread metrics table covering three dimensions:

  REACH    — how many tweets/users the cascade touched
  SPEED    — how quickly the cascade unfolded
  STRUCTURE — what shape the cascade took (broadcast vs. viral)

The unit of analysis is the thread (= source tweet + its reply tree).
Every metric is computed *per thread*, then joined back to threads_df
in the Phase 2 notebook.

KEY DESIGN DECISIONS

1. Tree topology comes from edges_df (which we built from structure.json).
   We trust the structure tree as canonical. Tweet content/timestamps come
   from tweets_df, which has fewer rows because some replies in the tree
   don't have JSONs on disk.

2. Speed metrics need timestamps, so they're computed only over the subset
   of replies we have JSONs for. We flag this with `tweets_with_timestamps`
   so Phase 3 can decide how to handle threads with low coverage.

3. Structural virality follows Goel et al. (2016, Management Science):
   "structural virality of a cascade is the average distance between all
   pairs of nodes in the diffusion tree." It distinguishes broadcast
   cascades (low virality — one source to many) from viral cascades
   (high virality — long chains of person-to-person spread).

4. We compute Wiener index too. Wiener = sum of pairwise distances.
   Structural virality = Wiener / C(n,2). They're equivalent up to
   normalization, but presenting both is useful: Wiener captures total
   "spreading work," structural virality normalizes by size.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def build_thread_tree(thread_id: str, edges_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build the reply tree for one thread as a directed graph (parent -> child).
    Returns an empty DiGraph if the thread has no edges (source-only thread).
    """
    g = nx.DiGraph()
    sub = edges_df[edges_df["thread_id"] == thread_id]
    g.add_node(thread_id)  # ensure root is present even with 0 edges
    for _, row in sub.iterrows():
        g.add_edge(row["parent_id"], row["child_id"])
    return g


# ---------------------------------------------------------------------------
# Reach metrics
# ---------------------------------------------------------------------------

def reach_metrics(tree: nx.DiGraph, root: str, tweets_subset: pd.DataFrame) -> dict:
    """
    Reach metrics. Operate on the full tree (from structure.json), not just
    the tweets we have JSONs for. The exception is unique_users, which can
    only be computed from tweets we have user info for.

    cascade_size:
      Number of nodes in the tree (1 = source-only, no replies).

    max_depth:
      Longest path from root to any leaf, in edges. 0 = source-only.

    max_breadth:
      Largest count of nodes at any single depth level.

    unique_users:
      Distinct user_ids among tweets_subset (those we have JSONs for).
      Will be ≤ cascade_size.
    """
    cascade_size = tree.number_of_nodes()

    if cascade_size == 1:
        max_depth, max_breadth = 0, 1
    else:
        # BFS from root, recording depth of each node
        depths = nx.single_source_shortest_path_length(tree, root)
        max_depth = max(depths.values())
        # Breadth = max count of nodes at any depth
        depth_counts = pd.Series(list(depths.values())).value_counts()
        max_breadth = int(depth_counts.max())

    unique_users = tweets_subset["user_id"].dropna().nunique()

    return {
        "cascade_size": cascade_size,
        "max_depth": max_depth,
        "max_breadth": max_breadth,
        "unique_users": unique_users,
    }


# ---------------------------------------------------------------------------
# Speed metrics
# ---------------------------------------------------------------------------

def speed_metrics(tweets_subset: pd.DataFrame, source_time: pd.Timestamp) -> dict:
    """
    Speed metrics. Computed only on tweets we have timestamps for.

    time_to_first_reply_min:
      Minutes between source tweet and earliest reply. NaN if no replies
      have timestamps.

    time_to_half_cascade_min:
      Minutes from source to the moment the cascade reached 50% of its
      final (observed) size. Captures whether the cascade exploded fast
      or grew slowly. NaN if cascade is source-only.

    reply_velocity_first_hour:
      Number of replies in the first 60 minutes after the source tweet.
      A simple "burstiness" measure.

    tweets_with_timestamps:
      How many tweets in the subset have parseable timestamps. Useful
      for filtering low-coverage threads downstream.
    """
    if pd.isna(source_time) or len(tweets_subset) <= 1:
        return {
            "time_to_first_reply_min": np.nan,
            "time_to_half_cascade_min": np.nan,
            "reply_velocity_first_hour": 0,
            "tweets_with_timestamps": int(tweets_subset["created_at"].notna().sum()),
        }

    replies = tweets_subset[~tweets_subset["is_source"]].copy()
    replies = replies[replies["created_at"].notna()]

    if len(replies) == 0:
        return {
            "time_to_first_reply_min": np.nan,
            "time_to_half_cascade_min": np.nan,
            "reply_velocity_first_hour": 0,
            "tweets_with_timestamps": 1,  # just the source
        }

    deltas_min = (replies["created_at"] - source_time).dt.total_seconds() / 60.0
    deltas_min = deltas_min.sort_values()
    # Drop replies with timestamps BEFORE the source (clock skew, rare)
    deltas_min = deltas_min[deltas_min >= 0]

    if len(deltas_min) == 0:
        return {
            "time_to_first_reply_min": np.nan,
            "time_to_half_cascade_min": np.nan,
            "reply_velocity_first_hour": 0,
            "tweets_with_timestamps": int(tweets_subset["created_at"].notna().sum()),
        }

    half_idx = max(0, len(deltas_min) // 2 - 1)  # median reply time

    return {
        "time_to_first_reply_min": float(deltas_min.iloc[0]),
        "time_to_half_cascade_min": float(deltas_min.iloc[half_idx]),
        "reply_velocity_first_hour": int((deltas_min <= 60).sum()),
        "tweets_with_timestamps": int(tweets_subset["created_at"].notna().sum()),
    }


# ---------------------------------------------------------------------------
# Structure metrics
# ---------------------------------------------------------------------------

def structure_metrics(tree: nx.DiGraph, root: str) -> dict:
    """
    Structural metrics that distinguish cascade *shapes*.

    structural_virality:
      Goel et al. 2016 — average shortest-path distance between all pairs
      of nodes in the (undirected) cascade tree. NaN for size-1 cascades.
      For size-2: 1.0 by definition.
      Low values → broadcast (one source, many siblings).
      High values → viral (long chains).

    wiener_index:
      Sum of pairwise distances. Equivalent to structural_virality * C(n,2).
      Useful for "total spreading work" interpretation.

    broadcast_ratio:
      Fraction of all nodes that are direct children of the root.
      1.0 = pure broadcast (everyone replied to source).
      → 0 = pure chain (everyone replied to a reply).

    branching_factor_mean:
      Mean number of children among all non-leaf nodes.
      A simple richness-of-branching measure.
    """
    n = tree.number_of_nodes()
    if n == 1:
        return {
            "structural_virality": np.nan,
            "wiener_index": 0.0,
            "broadcast_ratio": np.nan,
            "branching_factor_mean": np.nan,
        }
    if n == 2:
        return {
            "structural_virality": 1.0,
            "wiener_index": 1.0,
            "broadcast_ratio": 1.0,
            "branching_factor_mean": 1.0,
        }

    # Structural virality requires undirected distances
    undirected = tree.to_undirected()

    # Wiener index = sum of all-pairs shortest path distances
    # nx.wiener_index does this cleanly; fall back to manual if not available
    try:
        wiener = float(nx.wiener_index(undirected))
    except Exception:
        wiener = 0.0
        for src, lengths in nx.all_pairs_shortest_path_length(undirected):
            wiener += sum(lengths.values())
        wiener /= 2  # we counted each pair twice

    n_pairs = n * (n - 1) / 2
    structural_virality = wiener / n_pairs

    # Broadcast ratio: direct children of root / non-root nodes.
    # Clip to [0, 1] defensively — real PHEME trees won't exceed 1, but
    # malformed inputs (self-loops, etc.) shouldn't propagate garbage.
    direct_children = tree.out_degree(root) if root in tree else 0
    broadcast_ratio = min(direct_children / (n - 1), 1.0)

    # Branching factor: avg out-degree of non-leaf nodes
    out_degs = [d for _, d in tree.out_degree() if d > 0]
    branching_factor_mean = float(np.mean(out_degs)) if out_degs else np.nan

    return {
        "structural_virality": float(structural_virality),
        "wiener_index": wiener,
        "broadcast_ratio": float(broadcast_ratio),
        "branching_factor_mean": branching_factor_mean,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_metrics(
    threads_df: pd.DataFrame,
    tweets_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    progress_every: int = 500,
) -> pd.DataFrame:
    """
    Compute all metrics for every thread. Returns a dataframe keyed by
    thread_id, with columns covering reach + speed + structure.
    """
    # Pre-group for speed: building per-thread sub-dataframes once is much
    # faster than filtering inside the loop on every call.
    tweets_by_thread = dict(list(tweets_df.groupby("thread_id")))
    edges_by_thread = dict(list(edges_df.groupby("thread_id")))

    rows = []
    n_threads = len(threads_df)
    for i, (_, t) in enumerate(threads_df.iterrows()):
        thread_id = t["thread_id"]
        sub_edges = edges_by_thread.get(thread_id, edges_df.iloc[0:0])
        sub_tweets = tweets_by_thread.get(thread_id, tweets_df.iloc[0:0])

        tree = nx.DiGraph()
        tree.add_node(thread_id)
        for _, e in sub_edges.iterrows():
            tree.add_edge(e["parent_id"], e["child_id"])

        reach = reach_metrics(tree, thread_id, sub_tweets)
        speed = speed_metrics(sub_tweets, t["source_created_at"])
        struct = structure_metrics(tree, thread_id)

        rows.append({"thread_id": thread_id, **reach, **speed, **struct})

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  ... {i + 1:>5}/{n_threads} threads")

    return pd.DataFrame(rows)
