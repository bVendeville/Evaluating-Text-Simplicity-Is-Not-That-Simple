#!/usr/bin/env python3
"""
Refresh the release repo's per-dataset pickles with LLM-as-judge scores.

This script reads pre-computed merged "all-scores" CSV files from a separate
scoring directory (passed via ``--dev-results-dir``) and extends each release
pickle in ``data/`` in-place with new ``metric_scores`` keys for the LLM
judges that appear in the paper:

    Claude(simp), Claude(simp+src), Claude(simp+ref), Claude(simp+src+ref),
    GPT-4o-mini, Gemma-4-E4B, Ministral-8B, Llama-3.1-8B,
    GPT-4o-mini (pairwise), Gemma-4-E4B (pairwise),
    Ministral-8B (pairwise), Llama-3.1-8B (pairwise)

Pairwise variants apply only to the four human-annotated datasets (SDA,
ST-sent, ST-para, D-Wiki). Length-residualized ARTS pickles get LR-versions
of every direct-scoring LLM column, derived from the base ARTS scores via
ordinary-least-squares residualization on word count (same procedure used
for the existing ``human`` column in those pickles).

It also writes ``data/pairwise_position_bias.json`` mapping each (model,
dataset) cell to the pairwise position-bias rate reported in Table 12.

Usage
-----
    python scripts/build_release_data.py \\
        --dev-results-dir /path/to/scoring-results

The ``--dev-results-dir`` directory is expected to contain:

    all_scores/external/scores_<DS>_<STAMP>.csv   (one per dataset, latest)
    llm_judge/manifest.jsonl                       (one line per scoring cell)

Intended for the maintainer to refresh data after re-scoring. Reviewers do
not need to run this script — the pickles ship pre-built.

This script is idempotent: re-running just overwrites the targeted keys.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Latest merged-CSV timestamp produced upstream. If the upstream stamp ever
# changes, only this constant needs updating.
CSV_STAMP = "20260522_165216"

# Display names match the paper's PDF (Table 3 row labels) and the existing
# ``Claude(simp)`` convention already present in the release pickles. The
# prompt variant is implicit per model — Haiku uses Stage-0 prompts; the
# other four use the Kreutz et al. (2024) prompts — so no v2 suffix appears
# in the column name.
COLUMN_MAP_DIRECT_FULL = {
    # human-annotated datasets get all four Haiku conditions
    "claude_simp":          "Claude(simp)",
    "claude_simp_src":      "Claude(simp+src)",
    "claude_simp_ref":      "Claude(simp+ref)",
    "claude_simp_src_ref":  "Claude(simp+src+ref)",
    # four non-Haiku models, output-only, Kreutz et al. prompts
    "gpt4o_simp_v2":        "GPT-4o-mini",
    "gemma4_simp_v2":       "Gemma-4-E4B",
    "ministral_simp_v2":    "Ministral-8B",
    "llama_simp_v2":        "Llama-3.1-8B",
}

COLUMN_MAP_PAIRWISE = {
    "gpt4o_simp_pairwise_v2":      "GPT-4o-mini (pairwise)",
    "gemma4_simp_pairwise_v2":     "Gemma-4-E4B (pairwise)",
    "ministral_simp_pairwise_v2":  "Ministral-8B (pairwise)",
    "llama_simp_pairwise_v2":      "Llama-3.1-8B (pairwise)",
}

# ARTS / LR-ARTS only support output_only direct scoring (no source / ref
# / pairwise rows in the paper).
COLUMN_MAP_DIRECT_ARTS = {
    "claude_simp":          "Claude(simp)",
    "gpt4o_simp_v2":        "GPT-4o-mini",
    "gemma4_simp_v2":       "Gemma-4-E4B",
    "ministral_simp_v2":    "Ministral-8B",
    "llama_simp_v2":        "Llama-3.1-8B",
}

# Dataset name → release-repo pickle slug.
HUMAN_DATASETS = {
    "SDA":      "sda",
    "ST-sent":  "st_sent",
    "ST-para":  "st_para",
    "D-Wiki":   "d_wiki",
}
ARTS_DATASETS = {
    "ARTS94":   "arts94",
    "ARTS300":  "arts300",
    "ARTS3k":   "arts3k",
}
LR_ARTS_DATASETS = {
    "LR-ARTS94":   ("lr_arts94",  "arts94"),
    "LR-ARTS300":  ("lr_arts300", "arts300"),
    "LR-ARTS3k":   ("lr_arts3k",  "arts3k"),
}

# Manifest model-id → release column-prefix (matches COLUMN_MAP_PAIRWISE).
MANIFEST_MODEL_TO_RELEASE = {
    "gpt-4o-mini":      "GPT-4o-mini (pairwise)",
    "gemma-4":          "Gemma-4-E4B (pairwise)",
    "ministral-8b":     "Ministral-8B (pairwise)",
    "llama-3.1-8b":     "Llama-3.1-8B (pairwise)",
}


# --------------------------------------------------------------------------
# Pickle helpers
# --------------------------------------------------------------------------

def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: dict, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


# --------------------------------------------------------------------------
# CSV loading + alignment
# --------------------------------------------------------------------------

def load_csv(dev_results: Path, display_name: str) -> pd.DataFrame:
    """Load the canonical merged CSV for ``display_name``."""
    path = dev_results / "all_scores" / "external" / f"scores_{display_name}_{CSV_STAMP}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing merged CSV: {path}. Re-export from the scoring pipeline."
        )
    return pd.read_csv(path)


def align_by_text(df: pd.DataFrame, pkl: dict, display_name: str) -> pd.DataFrame:
    """Align CSV rows to the pickle's row order by matching text content.

    Positional alignment is unsafe when the CSV has stray duplicate rows
    interleaved (ARTS3k has 2 extras at positions 1956-1957). We match
    each pickle row to a unique CSV row by text content using a stable
    FIFO scan so duplicates resolve in first-come order.

    For ARTS-family pickles the join key is ``texts['text']`` vs the CSV's
    ``text`` column. For human-annotated pickles (SDA/ST-*/D-Wiki) it's the
    ``(source, simplified)`` pair vs the CSV's ``(original, text)`` columns.

    Returns a new DataFrame of the same length as the pickle, with rows in
    pickle order. Raises if any pickle row can't be mapped.
    """
    n = pkl["n_samples"]
    texts = pkl.get("texts", {})

    # Build a deque of available CSV indices per join key.
    from collections import defaultdict
    if "simplified" in texts and "source" in texts:
        # Human-annotated dataset: join on (source, simplified). The dev CSV's
        # column names differ slightly per dataset (SDA uses original+simplified;
        # ARTS/text-only uses text). Probe both candidate column names.
        simp_col = "simplified" if "simplified" in df.columns else "text"
        src_col  = "original"   if "original"   in df.columns else "source"
        pkl_keys = list(zip(texts["source"], texts["simplified"]))
        csv_keys = list(zip(df[src_col].astype(str), df[simp_col].astype(str)))
    elif "text" in texts:
        # ARTS-family: single text column. The dev CSV may name it 'text' or 'simplified'.
        simp_col = "text" if "text" in df.columns else "simplified"
        pkl_keys = list(texts["text"])
        csv_keys = list(df[simp_col].astype(str))
    else:
        raise ValueError(f"{display_name}: pickle has no 'texts' field to align on")

    # noop kept for clarity; the real construction is in the if/elif above.
    available = defaultdict(list)
    for i, k in enumerate(csv_keys):
        available[k].append(i)

    indices: List[int] = []
    unmatched: List[int] = []
    for j, k in enumerate(pkl_keys):
        bucket = available.get(k)
        if bucket:
            indices.append(bucket.pop(0))
        else:
            unmatched.append(j)
    if unmatched:
        raise ValueError(
            f"{display_name}: {len(unmatched)} pickle rows have no matching CSV row "
            f"(by text key). First missing: pkl_keys[{unmatched[0]}]={pkl_keys[unmatched[0]]!r}"
        )

    extras = sum(len(v) for v in available.values())
    if extras > 0:
        print(f"  {display_name}: text-aligned (dropped {extras} stray CSV rows "
              f"not in pickle: {len(df)} → {n})")
    elif len(df) != n:
        print(f"  {display_name}: text-aligned ({len(df)} → {n})")

    aligned = df.iloc[indices].copy().reset_index(drop=True)
    if len(aligned) != n:
        raise AssertionError(
            f"{display_name}: alignment produced {len(aligned)} rows, expected {n}"
        )
    return aligned


# --------------------------------------------------------------------------
# Per-dataset extension
# --------------------------------------------------------------------------

def extend_human_annotated(
    pickle_path: Path,
    csv_path_loader,                    # callable: () -> pd.DataFrame
    display_name: str,
) -> dict:
    """Extend a pickle for a human-annotated dataset (SDA / ST / D-Wiki)."""
    pkl = load_pickle(pickle_path)
    n = pkl["n_samples"]
    df = align_by_text(csv_path_loader(), pkl, display_name)

    added: List[str] = []
    for csv_col, release_name in COLUMN_MAP_DIRECT_FULL.items():
        if csv_col not in df.columns:
            continue
        pkl["metric_scores"][release_name] = df[csv_col].to_numpy(dtype=float)
        added.append(release_name)

    for csv_col, release_name in COLUMN_MAP_PAIRWISE.items():
        if csv_col not in df.columns:
            continue
        vals = df[csv_col].to_numpy(dtype=float)
        # Sanity: a column of constant 5.5 means BT had zero comparisons and
        # the fallback uniform was emitted. Convert to NaN so the row is
        # rendered as "—" downstream (matches paper Table 12).
        finite = vals[np.isfinite(vals)]
        if finite.size > 0 and (finite.max() - finite.min()) < 1e-9 and abs(finite[0] - 5.5) < 1e-3:
            vals = np.full(n, np.nan)
        pkl["metric_scores"][release_name] = vals
        added.append(release_name)

    save_pickle(pkl, pickle_path)
    return {"added": added, "n_samples": n, "total_metrics": len(pkl["metric_scores"]) - 1}


def extend_arts(
    pickle_path: Path,
    csv_path_loader,
    display_name: str,
) -> dict:
    """Extend an ARTS-family pickle (only direct-scoring columns)."""
    pkl = load_pickle(pickle_path)
    n = pkl["n_samples"]
    df = align_by_text(csv_path_loader(), pkl, display_name)

    added: List[str] = []
    for csv_col, release_name in COLUMN_MAP_DIRECT_ARTS.items():
        if csv_col not in df.columns:
            continue
        pkl["metric_scores"][release_name] = df[csv_col].to_numpy(dtype=float)
        added.append(release_name)

    save_pickle(pkl, pickle_path)
    return {"added": added, "n_samples": n, "total_metrics": len(pkl["metric_scores"]) - 1}


def _residualize(x: np.ndarray, word_count: np.ndarray) -> np.ndarray:
    """OLS-residualize ``x`` against ``word_count``.

    Matches the procedure used in the release pipeline to build the
    ``human`` column for LR-ARTS pickles.
    """
    mask = np.isfinite(x) & np.isfinite(word_count)
    if mask.sum() < 3:
        return np.full_like(x, np.nan, dtype=float)
    reg = LinearRegression()
    reg.fit(word_count[mask].reshape(-1, 1), x[mask])
    out = np.full_like(x, np.nan, dtype=float)
    pred = reg.predict(word_count[mask].reshape(-1, 1))
    out[mask] = x[mask] - pred
    return out


def extend_lr_arts(
    lr_pickle_path: Path,
    arts_pickle_path: Path,
) -> dict:
    """Build LR-ARTS LLM columns by residualizing the base ARTS LLM
    columns against ``Word_Count``.
    """
    lr_pkl = load_pickle(lr_pickle_path)
    arts_pkl = load_pickle(arts_pickle_path)
    n = lr_pkl["n_samples"]

    if arts_pkl["n_samples"] != n:
        raise ValueError(
            f"LR/ARTS row mismatch for {lr_pickle_path.name}: "
            f"{arts_pkl['n_samples']} vs {n}"
        )

    wc = arts_pkl["metric_scores"].get("Word_Count")
    if wc is None:
        raise KeyError(f"{arts_pickle_path.name} lacks 'Word_Count' metric")

    added: List[str] = []
    for release_name in COLUMN_MAP_DIRECT_ARTS.values():
        if release_name not in arts_pkl["metric_scores"]:
            continue
        x = np.asarray(arts_pkl["metric_scores"][release_name], dtype=float)
        lr_pkl["metric_scores"][release_name] = _residualize(x, np.asarray(wc, dtype=float))
        added.append(release_name)

    save_pickle(lr_pkl, lr_pickle_path)
    return {"added": added, "n_samples": n, "total_metrics": len(lr_pkl["metric_scores"]) - 1}


# --------------------------------------------------------------------------
# Pairwise position-bias JSON
# --------------------------------------------------------------------------

def build_pairwise_position_bias(dev_results: Path, out_path: Path) -> None:
    """Read manifest.jsonl, keep latest v2 pairwise entry per (model, ds),
    and emit a compact JSON: {model_release_name: {dataset: bias_rate}}.
    """
    manifest = dev_results / "llm_judge" / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    latest: Dict[Tuple[str, str], dict] = {}
    with manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("strategy") != "pairwise":
                continue
            if r.get("prompt_version") != "v2":
                continue
            if r.get("dataset") not in HUMAN_DATASETS:
                continue
            key = (r["model"], r["dataset"])
            latest[key] = r

    out: Dict[str, Dict[str, float]] = {}
    for (model_id, dataset), r in latest.items():
        release_name = MANIFEST_MODEL_TO_RELEASE.get(model_id)
        if not release_name:
            continue
        out.setdefault(release_name, {})[dataset] = r.get("position_bias_rate")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    n_cells = sum(len(v) for v in out.values())
    print(f"  pairwise_position_bias.json: {n_cells} (model, dataset) cells across {len(out)} models")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh release pickles with LLM-as-judge scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Maintainer-only: re-export merged CSVs from the scoring "
               "pipeline first, then point --dev-results-dir at them.",
    )
    parser.add_argument(
        "--dev-results-dir",
        required=True,
        type=Path,
        help="Path to scoring-results directory containing "
             "all_scores/external/scores_<DS>_<STAMP>.csv and "
             "llm_judge/manifest.jsonl.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Override the release-repo data/ directory (default: ../data).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    dev_results = args.dev_results_dir.resolve()
    data_dir = args.data_dir.resolve()
    print(f"Reading from:  {dev_results}")
    print(f"Writing pickles in:  {data_dir}")

    # 1) Human-annotated (SDA, ST-sent, ST-para, D-Wiki).
    print("\nHuman-annotated datasets:")
    for display, slug in HUMAN_DATASETS.items():
        result = extend_human_annotated(
            data_dir / f"{slug}.pkl",
            lambda d=display: load_csv(dev_results, d),
            display,
        )
        print(f"  {display:8s}  +{len(result['added']):2d} LLM cols  total metrics now {result['total_metrics']}")

    # 2) ARTS variants.
    print("\nARTS datasets:")
    for display, slug in ARTS_DATASETS.items():
        result = extend_arts(
            data_dir / f"{slug}.pkl",
            lambda d=display: load_csv(dev_results, d),
            display,
        )
        print(f"  {display:8s}  +{len(result['added']):2d} LLM cols  total metrics now {result['total_metrics']}")

    # 3) LR-ARTS variants — residualize from base ARTS LLM scores.
    print("\nLR-ARTS datasets (LLM cols residualized from base ARTS):")
    for display, (lr_slug, arts_slug) in LR_ARTS_DATASETS.items():
        result = extend_lr_arts(
            data_dir / f"{lr_slug}.pkl",
            data_dir / f"{arts_slug}.pkl",
        )
        print(f"  {display:10s}  +{len(result['added']):2d} LLM cols  total metrics now {result['total_metrics']}")

    # 4) Pairwise position-bias JSON.
    print("\nPairwise position-bias JSON:")
    build_pairwise_position_bias(dev_results, data_dir / "pairwise_position_bias.json")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
