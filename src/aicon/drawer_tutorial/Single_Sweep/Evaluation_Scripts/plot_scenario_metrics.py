import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def read_trials(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT t.success, t.timesteps, t.anchor_error, t.metadata FROM trials t")
    rows = []
    for success, timesteps, anchor_error, metadata_json in cur.fetchall():
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except Exception:
            metadata = {}
        rows.append({"success": bool(success), "timesteps": timesteps, "anchor_error": anchor_error, "metadata": metadata})
    conn.close()
    return rows


def analyze_by_key(rows, key: str):
    groups = defaultdict(list)
    for r in rows:
        md = r.get("metadata") or {}
        if key not in md:
            continue
        val = md.get(key)
        # keep raw value, but convert floats/ints for sorting
        groups[val].append(r)

    results = []
    for val, items in groups.items():
        n = len(items)
        success_count = sum(1 for it in items if it.get("success"))
        # average timesteps computed over successful runs only (None if no successes)
        success_timesteps = [it.get("timesteps") for it in items if it.get("success") and it.get("timesteps") is not None]
        avg_timesteps = float(np.mean(success_timesteps)) if success_timesteps else None
        # average anchor error computed over all runs (successful and unsuccessful)
        anchor_errors = [it.get("anchor_error") for it in items if it.get("anchor_error") is not None]
        avg_anchor_error = float(np.mean(anchor_errors)) if anchor_errors else None
        success_rate = float(success_count) / float(n) if n > 0 else None
        results.append((val, success_rate, avg_timesteps, avg_anchor_error, n))

    # Try to sort results naturally: numeric first then fallback to str ordering
    def sort_key(tup):
        v = tup[0]
        try:
            return (0, float(v))
        except Exception:
            return (1, str(v))

    results.sort(key=sort_key)
    return results


def plot_results(results, key: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(r[0]) for r in results]
    success_rates = [r[1] if r[1] is not None else 0.0 for r in results]
    avg_timesteps = [r[2] if r[2] is not None else np.nan for r in results]
    avg_anchor_errors = [r[3] if r[3] is not None else np.nan for r in results]

    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    # Success rate as a line plot
    color = "tab:blue"
    (h1,) = ax1.plot(x, success_rates, marker="o", linestyle="-", color=color, label="Success Rate")
    ax1.set_ylabel("Success Rate", color=color)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    # Show only every Nth label to avoid overcrowding. Target ~12 labels.
    max_labels = 12
    if len(labels) > max_labels:
        interval = int(np.ceil(len(labels) / max_labels))
    else:
        interval = 1
    display_labels = [lbl if (i % interval == 0) else "" for i, lbl in enumerate(labels)]
    ax1.set_xticklabels(display_labels, rotation=45, ha="right")
    ax1.tick_params(axis="y", labelcolor=color)

    # Average timesteps on secondary axis as a line plot
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    (h2,) = ax2.plot(x, avg_timesteps, marker="s", linestyle="-", color=color2, label="Avg Timesteps (success)")
    ax2.set_ylabel("Avg Timesteps (successful runs)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Anchor error on third axis for proper scaling
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    color3 = "tab:green"
    (h3,) = ax3.plot(x, avg_anchor_errors, marker="^", linestyle="-", color=color3, label="Avg Anchor Error (all)")
    ax3.set_ylabel("Avg Anchor Error (all runs)", color=color3)
    ax3.tick_params(axis="y", labelcolor=color3)

    # Combined legend
    handles = [h1, h2, h3]
    labels_legend = [h.get_label() for h in handles]
    ax1.legend(handles, labels_legend, loc="upper left")

    plt.title(f"Metrics by {key}")
    fig.tight_layout()
    out_file = out_dir / f"{key}_metrics.png"
    plt.savefig(out_file)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot success rate and average timesteps by metadata key values.")
    parser.add_argument("--db-path", required=True, help="Path to experiment_store.db")
    parser.add_argument("--out-dir", default="plots", help="Output directory for plots")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["disturbance", "noise_scale", "prior_noise_std"],
        help="Metadata keys to analyze (default: disturbance noise_scale prior_noise_std)",
    )

    args = parser.parse_args()
    db_path = Path(args.db_path)
    out_dir = Path(args.out_dir)

    rows = read_trials(db_path)
    if not rows:
        raise SystemExit(f"No trials found in {db_path}")

    for key in args.keys:
        results = analyze_by_key(rows, key)
        if not results:
            print(f"No metadata values found for key: {key}")
            continue
        plot_results(results, key, out_dir)
        print(f"Saved plot for {key} -> {out_dir / (key + '_metrics.png')}")


if __name__ == "__main__":
    main()
