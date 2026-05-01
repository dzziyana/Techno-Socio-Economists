"""
PHEME dataset loader.

Walks the PHEME folder structure and builds three pandas DataFrames:
  - threads_df: one row per source tweet (the unit of analysis)
  - tweets_df:  one row per tweet (source + every reply), keyed to thread_id
  - edges_df:   one row per parent->child reply edge (for graph construction)

Expected folder layout (PHEME-9 "all-rnr-annotated-threads"):

    root/
      <event>-all-rnr-threads/
        rumours/
          <thread_id>/
            annotation.json
            structure.json
            source-tweets/<thread_id>.json
            reactions/<reply_id>.json (many)
        non-rumours/
          <thread_id>/
            ...

Notes on the data:
  - `annotation.json` for non-rumours: {"is_rumour": "nonrumour"}.
  - `annotation.json` for rumours: includes a "category" or "true"/"false"/
    "unverified" field — naming varies slightly across PHEME versions, so we
    normalize it.
  - `structure.json` is a nested dict: keys are tweet ids (as strings),
    values are either [] (no replies) or another dict of children. The root
    key equals the source tweet id.
  - Reply tweet objects have `in_reply_to_status_id` pointing at the parent.
    We trust `structure.json` as the canonical tree, but cross-check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


# ---------------------------------------------------------------------------
# Veracity normalization
# ---------------------------------------------------------------------------

def _normalize_veracity(annotation: dict) -> tuple[str, str]:
    """
    Returns (is_rumour, veracity) where:
      is_rumour  ∈ {"rumour", "nonrumour"}
      veracity   ∈ {"true", "false", "unverified", "nonrumour"}

    PHEME annotation files are inconsistent across events. We handle:
      - {"is_rumour": "nonrumour"}                        → ("nonrumour", "nonrumour")
      - {"is_rumour": "rumour", "category": "true"}       → ("rumour", "true")
      - {"is_rumour": "rumour", "true": "1"}              → ("rumour", "true")
      - {"is_rumour": "rumour", "misinformation": "0"}    → ("rumour", "true")  (legacy)
    """
    is_rumour = annotation.get("is_rumour", "").lower()

    if is_rumour == "nonrumour":
        return ("nonrumour", "nonrumour")

    # Rumour: try to find a veracity label
    if "category" in annotation:
        return ("rumour", annotation["category"].lower())

    for label in ("true", "false", "unverified"):
        if label in annotation and str(annotation[label]) == "1":
            return ("rumour", label)

    # PHEME-5 legacy: "misinformation" flag (1 = false rumour, 0 = true rumour)
    if "misinformation" in annotation:
        return ("rumour", "false" if str(annotation["misinformation"]) == "1" else "true")

    return ("rumour", "unverified")


# ---------------------------------------------------------------------------
# Tree flattening
# ---------------------------------------------------------------------------

def _walk_structure(
    node: dict,
    parent_id: str | None = None,
    skip_counter: list | None = None,
) -> Iterator[tuple[str, str | None]]:
    """
    Yields (tweet_id, parent_id) pairs from a structure.json tree.
    Root tweet has parent_id=None.

    PHEME's structure.json occasionally contains non-numeric placeholder
    keys (e.g. empty strings, the word "missing") representing replies
    whose tweet ID couldn't be resolved at collection time. We drop
    those nodes AND their descendants — we can't trust the parent-child
    chain past an unidentifiable node.

    `skip_counter` is an optional one-element list used by the caller to
    count how many nodes we dropped (cheap mutable accumulator across
    recursion).
    """
    if not isinstance(node, dict):
        return
    for tweet_id, children in node.items():
        if not (isinstance(tweet_id, str) and tweet_id.isdigit()):
            # Malformed key: skip this subtree entirely.
            if skip_counter is not None:
                skip_counter[0] += 1
            continue
        yield tweet_id, parent_id
        if isinstance(children, dict):
            yield from _walk_structure(children, parent_id=tweet_id, skip_counter=skip_counter)
        # if children is [] we just stop — leaf node


# ---------------------------------------------------------------------------
# Per-tweet feature extraction
# ---------------------------------------------------------------------------

def _extract_tweet_fields(tweet: dict) -> dict:
    """Pull the fields we'll actually use into a flat dict. Keep it minimal."""
    user = tweet.get("user") or {}
    return {
        "tweet_id": str(tweet["id"]),
        "created_at": tweet.get("created_at"),
        "text": tweet.get("text", ""),
        "lang": tweet.get("lang"),
        "favorite_count": tweet.get("favorite_count", 0),
        "retweet_count": tweet.get("retweet_count", 0),
        "in_reply_to_status_id": (
            str(tweet["in_reply_to_status_id"])
            if tweet.get("in_reply_to_status_id") is not None
            else None
        ),
        # User features — useful for "who spreads what"
        "user_id": str(user.get("id")) if user.get("id") is not None else None,
        "user_screen_name": user.get("screen_name"),
        "user_followers_count": user.get("followers_count", 0),
        "user_friends_count": user.get("friends_count", 0),
        "user_statuses_count": user.get("statuses_count", 0),
        "user_verified": bool(user.get("verified", False)),
        "user_created_at": user.get("created_at"),
    }


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

@dataclass
class PhemeLoader:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        if not self.root.exists():
            raise FileNotFoundError(f"PHEME root not found: {self.root}")

    def iter_threads(self) -> Iterator[Path]:
        """Yield every thread folder (one per source tweet)."""
        for event_dir in sorted(self.root.iterdir()):
            if not event_dir.is_dir() or not event_dir.name.endswith("-all-rnr-threads"):
                continue
            for label_dir_name in ("rumours", "non-rumours"):
                label_dir = event_dir / label_dir_name
                if not label_dir.exists():
                    continue
                for thread_dir in sorted(label_dir.iterdir()):
                    if thread_dir.is_dir():
                        yield thread_dir

    def parse_thread(self, thread_dir: Path) -> dict | None:
        """
        Parse one thread folder. Returns a dict with three lists/values:
          - thread_row: dict for threads_df
          - tweet_rows: list of dicts for tweets_df
          - edge_rows:  list of dicts for edges_df
        Returns None if the thread is malformed (missing source, etc.).
        """
        thread_id = thread_dir.name
        event = thread_dir.parent.parent.name.replace("-all-rnr-threads", "")

        # --- annotation
        annotation_path = thread_dir / "annotation.json"
        if not annotation_path.exists():
            return None
        with annotation_path.open() as f:
            annotation = json.load(f)
        is_rumour, veracity = _normalize_veracity(annotation)

        # --- source tweet
        source_path = thread_dir / "source-tweets" / f"{thread_id}.json"
        if not source_path.exists():
            return None
        with source_path.open() as f:
            source_tweet = json.load(f)
        source_fields = _extract_tweet_fields(source_tweet)

        # --- reactions
        reactions_dir = thread_dir / "reactions"
        reaction_tweets = {}
        if reactions_dir.exists():
            for reply_path in reactions_dir.glob("*.json"):
                try:
                    with reply_path.open() as f:
                        reply = json.load(f)
                    reaction_tweets[str(reply["id"])] = reply
                except (json.JSONDecodeError, KeyError):
                    continue  # skip corrupt reply files; very rare

        # --- structure
        structure_path = thread_dir / "structure.json"
        edges = []
        all_tweet_ids_in_tree = set()
        n_skipped_nodes = 0
        if structure_path.exists():
            with structure_path.open() as f:
                structure = json.load(f)
            skip_counter = [0]
            for tid, pid in _walk_structure(structure, skip_counter=skip_counter):
                all_tweet_ids_in_tree.add(tid)
                if pid is not None:  # skip the root (no parent)
                    edges.append({"thread_id": thread_id, "parent_id": pid, "child_id": tid})
            n_skipped_nodes = skip_counter[0]

        # --- assemble tweet rows (source + reactions)
        tweet_rows = [{**source_fields, "thread_id": thread_id, "is_source": True}]
        for tid, reply in reaction_tweets.items():
            row = _extract_tweet_fields(reply)
            row["thread_id"] = thread_id
            row["is_source"] = False
            tweet_rows.append(row)

        # --- thread-level row
        thread_row = {
            "thread_id": thread_id,
            "event": event,
            "is_rumour": is_rumour,
            "veracity": veracity,
            "source_user_id": source_fields["user_id"],
            "source_user_screen_name": source_fields["user_screen_name"],
            "source_user_followers": source_fields["user_followers_count"],
            "source_user_verified": source_fields["user_verified"],
            "source_created_at": source_fields["created_at"],
            "source_text": source_fields["text"],
            "source_retweet_count": source_fields["retweet_count"],
            "source_favorite_count": source_fields["favorite_count"],
            "n_reactions": len(reaction_tweets),
            "n_edges": len(edges),
            "n_tweets_in_structure": len(all_tweet_ids_in_tree),
            "n_skipped_malformed_nodes": n_skipped_nodes,
        }

        return {
            "thread_row": thread_row,
            "tweet_rows": tweet_rows,
            "edge_rows": edges,
        }

    def load_all(self, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Walk everything and return (threads_df, tweets_df, edges_df)."""
        thread_rows, tweet_rows, edge_rows = [], [], []
        n_ok, n_skip = 0, 0
        for thread_dir in self.iter_threads():
            parsed = self.parse_thread(thread_dir)
            if parsed is None:
                n_skip += 1
                continue
            thread_rows.append(parsed["thread_row"])
            tweet_rows.extend(parsed["tweet_rows"])
            edge_rows.extend(parsed["edge_rows"])
            n_ok += 1

        if verbose:
            print(f"Parsed {n_ok} threads, skipped {n_skip} (malformed/missing).")
            if thread_rows:
                total_skipped_nodes = sum(r.get("n_skipped_malformed_nodes", 0) for r in thread_rows)
                affected_threads = sum(1 for r in thread_rows if r.get("n_skipped_malformed_nodes", 0) > 0)
                if total_skipped_nodes > 0:
                    print(
                        f"Dropped {total_skipped_nodes} malformed nodes "
                        f"(non-numeric IDs in structure.json) "
                        f"across {affected_threads} threads."
                    )

        threads_df = pd.DataFrame(thread_rows)
        tweets_df = pd.DataFrame(tweet_rows)
        edges_df = pd.DataFrame(edge_rows)

        # Parse Twitter timestamps once, here, so downstream code doesn't repeat it.
        # Twitter format: "Wed Jan 07 11:11:33 +0000 2015"
        for df, col in [
            (threads_df, "source_created_at"),
            (tweets_df, "created_at"),
        ]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)

        return threads_df, tweets_df, edges_df
