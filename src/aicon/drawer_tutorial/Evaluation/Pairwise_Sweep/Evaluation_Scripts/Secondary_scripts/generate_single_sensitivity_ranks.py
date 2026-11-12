#!/usr/bin/env python3
import csv
from pathlib import Path


def dense_ranks(values):
    # values: list of floats, return dict value->rank (1 = highest)
    unique = sorted(set(values), reverse=True)
    rank_map = {v: i+1 for i, v in enumerate(unique)}
    return rank_map


def main():
    base = Path(__file__).resolve().parent.parent.parent / "data" / "scenario_interaction_rank_analysis"
    src = base / "single_sensitivity_scenario_summary.csv"
    dst = base / "scenario_parameter_ranks_single_sensitivity.csv"
    if not src.exists():
        print(f"Source file not found: {src}")
        return 1

    with open(src, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    scenarios = [c for c in reader.fieldnames if c != 'parameter' and c != 'max_sensitivity']
    parameters = [r['parameter'] for r in rows]

    # Build per-scenario value lists
    per_scenario_values = {s: [] for s in scenarios}
    for r in rows:
        for s in scenarios:
            try:
                v = float(r.get(s, '') or 'nan')
            except Exception:
                v = float('nan')
            per_scenario_values[s].append(v)

    # Compute ranks per scenario (dense ranks, 1 = highest)
    per_scenario_ranks = {s: [] for s in scenarios}
    for s in scenarios:
        vals = per_scenario_values[s]
        # handle NaNs by treating them as lowest
        nan_mask = [v != v for v in vals]
        valid_vals = [v for v in vals if v == v]
        rank_map = dense_ranks(valid_vals)
        for v in vals:
            if v != v:
                per_scenario_ranks[s].append(len(valid_vals) + 1)
            else:
                per_scenario_ranks[s].append(rank_map.get(v, len(valid_vals) + 1))

    # Write output CSV matching other rank files format
    with open(dst, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['parameter'] + scenarios)
        for i, p in enumerate(parameters):
            row = [p] + [per_scenario_ranks[s][i] for s in scenarios]
            writer.writerow(row)

    print(f"Wrote ranks to {dst}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
