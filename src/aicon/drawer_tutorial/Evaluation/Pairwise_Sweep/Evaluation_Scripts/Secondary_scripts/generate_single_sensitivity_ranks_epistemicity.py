#!/usr/bin/env python3
import csv
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = [("Normal", "normal"), ("Disturbance", "disturbance"), ("Bad Prior", "bad_prior"), ("Noise", "noise")]
OUT_DIR = PROJECT_ROOT / "data" / "scenario_interaction_rank_analysis_epistemicity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load single_sensitivity per scenario
scenario_values = {}
for display, folder in SCENARIOS:
    path = PROJECT_ROOT / "data" / folder / "metrics" / "epistemicity" / "si_vs_residuals_summary.csv"
    values = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                param = row.get("parameter", "").strip()
                if not param:
                    continue
                try:
                    v = float(row.get("single_sensitivity", ""))
                except Exception:
                    v = float('nan')
                if math.isfinite(v):
                    values[param] = v
    scenario_values[display] = values

# Find common params
sets = [set(vals.keys()) for vals in scenario_values.values()]
if sets:
    common = set.intersection(*sets)
else:
    common = set()

ordered_params = sorted(common)

# compute ranks (higher value -> rank 1)
def compute_rank_vector(score_map):
    # score_map: param->value
    items = [(p, v) for p, v in score_map.items() if p in ordered_params]
    # sort descending by value
    items_sorted = sorted(items, key=lambda kv: kv[1], reverse=True)
    ranks = {p: rank for rank, (p, _) in enumerate(items_sorted, start=1)}
    return ranks

rank_table = {display: compute_rank_vector(vals) for display, vals in scenario_values.items()}

out_path = OUT_DIR / "scenario_parameter_ranks_single_sensitivity.csv"
with out_path.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["parameter", *[d for d, _ in SCENARIOS]])
    for param in ordered_params:
        row = [param]
        for display, _ in SCENARIOS:
            row.append(rank_table.get(display, {}).get(param, float('nan')))
        writer.writerow(row)

print(f"Wrote single sensitivity ranks to {out_path}")
