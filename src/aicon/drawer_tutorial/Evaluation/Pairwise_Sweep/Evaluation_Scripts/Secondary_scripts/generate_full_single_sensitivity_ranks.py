#!/usr/bin/env python3
import csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_SCORES = {
    "Normal": PROJECT_ROOT / "data" / "normal" / "metrics" / "success" / "si_vs_residuals_summary.csv",
    "Disturbance": PROJECT_ROOT / "data" / "disturbance" / "metrics" / "success" / "si_vs_residuals_summary.csv",
    "Bad Prior": PROJECT_ROOT / "data" / "bad_prior" / "metrics" / "success" / "si_vs_residuals_summary.csv",
    "Noise": PROJECT_ROOT / "data" / "noise" / "metrics" / "success" / "si_vs_residuals_summary.csv",
}
SENSITIVITY_MEASURE = "single_sensitivity"


def load_scores():
    scenario_maps = {}
    for scenario, path in SCENARIO_SCORES.items():
        scores = {}
        if not path.exists():
            scenario_maps[scenario] = scores
            continue
        with path.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                param = str(row.get('parameter','')).strip()
                if not param:
                    continue
                try:
                    v = float(row.get(SENSITIVITY_MEASURE, '') or 'nan')
                except Exception:
                    v = float('nan')
                scores[param] = v
        scenario_maps[scenario] = scores
    return scenario_maps


def dense_rank_list(values):
    # values: list of floats
    valid = [v for v in values if not (v != v)]
    unique = sorted(set(valid), reverse=True)
    rank_map = {v: i+1 for i, v in enumerate(unique)}
    ranks = []
    for v in values:
        if v != v:
            ranks.append(len(valid) + 1)
        else:
            ranks.append(rank_map.get(v, len(valid) + 1))
    return ranks


def main():
    scenario_maps = load_scores()
    all_params = sorted({p for scores in scenario_maps.values() for p in scores.keys()})
    scenarios = list(SCENARIO_SCORES.keys())

    per_scenario_values = {s: [scenario_maps[s].get(p, float('nan')) for p in all_params] for s in scenarios}

    per_scenario_ranks = {s: dense_rank_list(per_scenario_values[s]) for s in scenarios}

    out = PROJECT_ROOT / 'data' / 'scenario_interaction_rank_analysis' / 'scenario_parameter_ranks_single_sensitivity_full.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['parameter'] + scenarios)
        for i, p in enumerate(all_params):
            row = [p] + [per_scenario_ranks[s][i] for s in scenarios]
            writer.writerow(row)

    print(f'Wrote full ranks to {out}')


if __name__ == '__main__':
    raise SystemExit(main())
