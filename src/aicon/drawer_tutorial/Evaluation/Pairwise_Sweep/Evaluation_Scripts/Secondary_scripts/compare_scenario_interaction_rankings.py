from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_NAMES = ["Normal", "Disturbance", "Bad Prior", "Noise"]

def load_scenario_scores(metric: str = "success") -> dict[str, dict[str, float]]:
    scenario_maps: dict[str, dict[str, float]] = {}
    for scenario in SCENARIO_NAMES:
        # map display name to folder name (e.g. 'Bad Prior' -> 'bad_prior')
        folder = scenario.lower().replace(" ", "_")
        path = PROJECT_ROOT / "data" / folder / "metrics" / metric / "si_vs_residuals_summary.csv"
        scores: dict[str, float] = {}
        if not path.exists():
            scenario_maps[scenario] = scores
            continue

        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                parameter = str(row.get("parameter", "")).strip()
                if not parameter:
                    continue
                parameter_scores = {}
                for measure in [SENSITIVITY_MEASURE, *MEASURES]:
                    value = row.get(measure, "")
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        numeric = np.nan
                    parameter_scores[measure] = numeric
                scores[parameter] = parameter_scores
        scenario_maps[scenario] = scores
    return scenario_maps
MEASURES = ["mean_interaction", "max_interaction"]
SENSITIVITY_MEASURE = "single_sensitivity"





def get_common_param_set(scenario_maps: dict[str, dict[str, float]], measure: str) -> set[str]:
    valid_by_scenario = {
        scenario: {
            parameter
            for parameter, values in params.items()
            if isinstance(values, dict) and np.isfinite(values.get(measure, np.nan))
        }
        for scenario, params in scenario_maps.items()
    }
    if not valid_by_scenario:
        return set()
    return set.intersection(*valid_by_scenario.values())


def compute_pairwise_rho(values_a: dict[str, float], values_b: dict[str, float]) -> tuple[float, float]:
    shared = sorted(set(values_a) & set(values_b))
    left = []
    right = []
    for parameter in shared:
        va = values_a.get(parameter, np.nan)
        vb = values_b.get(parameter, np.nan)
        if np.isfinite(va) and np.isfinite(vb):
            left.append(float(va))
            right.append(float(vb))

    if len(left) < 2:
        return np.nan, np.nan

    rho, p_value = spearmanr(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    return float(rho), float(p_value)


def compute_rank_vector(score_map: dict[str, float]) -> dict[str, float]:
    valid_scores = {parameter: value for parameter, value in score_map.items() if np.isfinite(value)}
    ranked = sorted(valid_scores.items(), key=lambda kv: kv[1], reverse=True)
    ranks = {parameter: rank for rank, (parameter, _) in enumerate(ranked, start=1)}
    return ranks


def compute_mean_max_spearman_by_scenario(scenario_maps: dict[str, dict[str, float]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for scenario_name, score_map in scenario_maps.items():
        valid_params = sorted(
            parameter
            for parameter, values in score_map.items()
            if isinstance(values, dict)
            and np.isfinite(values.get("mean_interaction", np.nan))
            and np.isfinite(values.get("max_interaction", np.nan))
        )
        if len(valid_params) < 2:
            rows.append({"scenario": scenario_name, "rho": np.nan, "p_value": np.nan, "n_params": len(valid_params)})
            continue

        mean_values = np.asarray([score_map[p]["mean_interaction"] for p in valid_params], dtype=float)
        max_values = np.asarray([score_map[p]["max_interaction"] for p in valid_params], dtype=float)
        mean_ranks = rankdata(mean_values, method="average")
        max_ranks = rankdata(max_values, method="average")
        rho, p_value = spearmanr(mean_ranks, max_ranks)
        rows.append({"scenario": scenario_name, "rho": float(rho), "p_value": float(p_value), "n_params": len(valid_params)})

    return rows


def compute_sensitivity_max_spearman_by_scenario(scenario_maps: dict[str, dict[str, float]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for scenario_name, score_map in scenario_maps.items():
        valid_params = sorted(
            parameter
            for parameter, values in score_map.items()
            if isinstance(values, dict)
            and np.isfinite(values.get(SENSITIVITY_MEASURE, np.nan))
            and np.isfinite(values.get("max_interaction", np.nan))
        )
        if len(valid_params) < 2:
            rows.append({"scenario": scenario_name, "rho": np.nan, "p_value": np.nan, "n_params": len(valid_params)})
            continue

        sensitivity_values = np.asarray([score_map[p][SENSITIVITY_MEASURE] for p in valid_params], dtype=float)
        max_values = np.asarray([score_map[p]["max_interaction"] for p in valid_params], dtype=float)
        sensitivity_ranks = rankdata(sensitivity_values, method="average")
        max_ranks = rankdata(max_values, method="average")
        rho, p_value = spearmanr(sensitivity_ranks, max_ranks)
        rows.append({"scenario": scenario_name, "rho": float(rho), "p_value": float(p_value), "n_params": len(valid_params)})

    return rows


def build_spearman_summary(scenario_maps: dict[str, dict[str, float]], measure: str) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    scenario_names = list(scenario_maps.keys())
    rho_matrix = np.eye(len(scenario_names), dtype=float)
    p_matrix = np.zeros((len(scenario_names), len(scenario_names)), dtype=float)
    rank_table: dict[str, dict[str, float]] = {}

    all_valid_params = get_common_param_set(scenario_maps, measure)
    for scenario_name in scenario_names:
        scenario_values = scenario_maps[scenario_name]
        rank_table[scenario_name] = {
            parameter: compute_rank_vector({p: scenario_values.get(p, {}).get(measure, np.nan) for p in all_valid_params})[parameter]
            for parameter in all_valid_params
            if parameter in scenario_values and np.isfinite(scenario_values.get(parameter, {}).get(measure, np.nan))
        }

    for i, first in enumerate(scenario_names):
        for j in range(i + 1, len(scenario_names)):
            second = scenario_names[j]
            rho, p_value = compute_pairwise_rho(
                {parameter: values.get(measure, np.nan) for parameter, values in scenario_maps[first].items()},
                {parameter: values.get(measure, np.nan) for parameter, values in scenario_maps[second].items()},
            )
            rho_matrix[i, j] = rho
            rho_matrix[j, i] = rho
            p_matrix[i, j] = p_value
            p_matrix[j, i] = p_value

    for idx, scenario_name in enumerate(scenario_names):
        p_matrix[idx, idx] = 0.0

    return rho_matrix, p_matrix, rank_table


def write_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scenario", *labels])
        for i, label in enumerate(labels):
            writer.writerow([label, *[float(val) for val in matrix[i, :]]])


def write_rank_table_csv(path: Path, rank_table: dict[str, dict[str, float]], ordered_params: list[str], scenario_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["parameter", *scenario_names])
        for parameter in ordered_params:
            row = [parameter]
            for scenario_name in scenario_names:
                row.append(rank_table.get(scenario_name, {}).get(parameter, np.nan))
            writer.writerow(row)


def save_heatmap(matrix: np.ndarray, labels: list[str], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", color="black" if abs(value) < 0.6 else "white")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def compute_single_sensitivity_scenario_spearman(scenario_maps: dict[str, dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    scenario_names = list(scenario_maps.keys())
    rho_matrix = np.eye(len(scenario_names), dtype=float)
    p_matrix = np.zeros((len(scenario_names), len(scenario_names)), dtype=float)

    for i, first in enumerate(scenario_names):
        for j in range(i + 1, len(scenario_names)):
            second = scenario_names[j]
            common = sorted(
                set(
                    parameter
                    for parameter, values in scenario_maps[first].items()
                    if isinstance(values, dict) and np.isfinite(values.get(SENSITIVITY_MEASURE, np.nan))
                )
                & set(
                    parameter
                    for parameter, values in scenario_maps[second].items()
                    if isinstance(values, dict) and np.isfinite(values.get(SENSITIVITY_MEASURE, np.nan))
                )
            )
            if len(common) < 2:
                rho = np.nan
                p_value = np.nan
            else:
                left = np.asarray([scenario_maps[first][parameter][SENSITIVITY_MEASURE] for parameter in common], dtype=float)
                right = np.asarray([scenario_maps[second][parameter][SENSITIVITY_MEASURE] for parameter in common], dtype=float)
                rho, p_value = spearmanr(left, right)
            rho_matrix[i, j] = rho
            rho_matrix[j, i] = rho
            p_matrix[i, j] = p_value
            p_matrix[j, i] = p_value

    for idx in range(len(scenario_names)):
        p_matrix[idx, idx] = 0.0
    return rho_matrix, p_matrix


def save_measure_outputs(measure: str, scenario_maps: dict[str, dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_names = list(scenario_maps.keys())
    rho_matrix, p_matrix, rank_table = build_spearman_summary(scenario_maps, measure)

    all_valid_params = get_common_param_set(scenario_maps, measure)
    ordered_params = sorted(all_valid_params)
    write_matrix_csv(out_dir / f"spearman_rho_{measure}.csv", rho_matrix, scenario_names)
    write_matrix_csv(out_dir / f"spearman_p_value_{measure}.csv", p_matrix, scenario_names)
    write_rank_table_csv(out_dir / f"scenario_parameter_ranks_{measure}.csv", rank_table, ordered_params, scenario_names)
    save_heatmap(rho_matrix, scenario_names, out_dir / f"spearman_rho_heatmap_{measure}.png", f"Spearman rho matrix ({measure})")

    print(f"\n=== {measure} ===")
    print("Scenario names:", scenario_names)
    print("Common parameters:", len(ordered_params))
    print("Rho matrix:")
    print(rho_matrix)
    print("P-value matrix:")
    print(p_matrix)
    print("Rank table:")
    for param in ordered_params:
        print(param, {scenario: rank_table.get(scenario, {}).get(param, np.nan) for scenario in scenario_names})

    pair_rows = []
    for i, first in enumerate(scenario_names):
        for j in range(i + 1, len(scenario_names)):
            second = scenario_names[j]
            pair_rows.append(
                {
                    "measure": measure,
                    "scenario_a": first,
                    "scenario_b": second,
                    "rho": float(rho_matrix[i, j]),
                    "p_value": float(p_matrix[i, j]),
                }
            )
    with (out_dir / f"spearman_pair_summary_{measure}.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["measure", "scenario_a", "scenario_b", "rho", "p_value"])
        writer.writeheader()
        writer.writerows(pair_rows)


def build_sensitivity_heatmap_outputs(scenario_maps: dict[str, dict[str, float]], out_dir: Path, top_n: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_order = ["Normal", "Disturbance", "Bad Prior", "Noise"]
    scenario_values = {
        scenario: {
            parameter: float(values.get(SENSITIVITY_MEASURE, np.nan))
            for parameter, values in scenario_maps.get(scenario, {}).items()
            if isinstance(values, dict) and np.isfinite(values.get(SENSITIVITY_MEASURE, np.nan))
        }
        for scenario in scenario_order
    }

    all_params = sorted({p for values in scenario_values.values() for p in values})
    max_by_param = {
        parameter: max(
            [
                float(scenario_values[scenario].get(parameter, np.nan))
                for scenario in scenario_order
                if np.isfinite(scenario_values[scenario].get(parameter, np.nan))
            ],
            default=np.nan,
        )
        for parameter in all_params
    }
    if top_n is None:
        # Use all parameters
        selected = sorted(all_params, key=lambda p: max_by_param.get(p, np.nan), reverse=True)
    else:
        selected = [
            parameter
            for parameter, _ in sorted(max_by_param.items(), key=lambda item: item[1], reverse=True)[:top_n]
        ]

    matrix = np.full((len(selected), len(scenario_order)), np.nan, dtype=float)
    for row_idx, parameter in enumerate(selected):
        for col_idx, scenario in enumerate(scenario_order):
            value = scenario_values[scenario].get(parameter, np.nan)
            if np.isfinite(value):
                matrix[row_idx, col_idx] = float(value)

    vmax = float(np.nanmax(matrix)) if np.isfinite(np.nanmax(matrix)) else 1.0
    vmin = 0.0

    fig, ax = plt.subplots(figsize=(8, 9))
    masked_matrix = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked_matrix, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(scenario_order)))
    ax.set_xticklabels(scenario_order, rotation=30, ha="right")
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected, fontsize=8)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Parameter")
    ax.set_title("Single-parameter sensitivity across scenarios")

    for idx, parameter in enumerate(selected):
        if parameter.startswith("connection.DistGraspHandConnection."):
            ax.get_yticklabels()[idx].set_color("#1d4ed8")
            ax.get_yticklabels()[idx].set_fontweight("bold")

    for row_idx, parameter in enumerate(selected):
        for col_idx, scenario in enumerate(scenario_order):
            value = matrix[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            text_color = "white" if value > vmax * 0.5 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}" if value < 1.0 else f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Single sensitivity S_i")
    fig.tight_layout()
    fig.savefig(out_dir / "single_sensitivity_scenario_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "single_sensitivity_scenario_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for parameter in selected:
        row = {"parameter": parameter}
        for scenario in scenario_order:
            row[scenario] = float(scenario_values[scenario].get(parameter, np.nan)) if np.isfinite(scenario_values[scenario].get(parameter, np.nan)) else np.nan
        row["max_sensitivity"] = float(max_by_param.get(parameter, np.nan))
        summary_rows.append(row)

    with (out_dir / "single_sensitivity_scenario_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["parameter", *scenario_order, "max_sensitivity"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== Single sensitivity heatmap summary ===")
    for row in summary_rows:
        print(row)


def _run_analysis_for_metric(metric: str, out_dir: Path) -> None:
    scenario_maps = load_scenario_scores(metric=metric)
    out_dir = out_dir

    # Save mean/max spearman and rank tables
    for measure in MEASURES:
        save_measure_outputs(measure, scenario_maps, out_dir)

    # Single-sensitivity heatmap and spearman matrices
    build_sensitivity_heatmap_outputs(scenario_maps, out_dir)

    sensitivity_rho_matrix, sensitivity_p_matrix = compute_single_sensitivity_scenario_spearman(scenario_maps)
    scenario_names = list(scenario_maps.keys())
    write_matrix_csv(out_dir / "spearman_rho_single_sensitivity.csv", sensitivity_rho_matrix, scenario_names)
    write_matrix_csv(out_dir / "spearman_p_value_single_sensitivity.csv", sensitivity_p_matrix, scenario_names)
    save_heatmap(sensitivity_rho_matrix, scenario_names, out_dir / "spearman_rho_heatmap_single_sensitivity.png", f"Spearman rho matrix (single sensitivity: {metric})")

    print(f"\n=== Single sensitivity scenario correlation ({metric}) ===")
    print(sensitivity_rho_matrix)
    print("P-values:")

    # Write rank table CSV for single sensitivity across scenarios
    scenario_names = list(scenario_maps.keys())
    all_valid_params = get_common_param_set(scenario_maps, SENSITIVITY_MEASURE)
    ordered_params = sorted(all_valid_params)
    # Build rank table: scenario -> parameter -> rank
    rank_table: dict[str, dict[str, float]] = {}
    for scenario_name in scenario_names:
        values = scenario_maps.get(scenario_name, {})
        rank_table[scenario_name] = {
            parameter: compute_rank_vector({p: values.get(p, {}).get(SENSITIVITY_MEASURE, np.nan) for p in all_valid_params})[parameter]
            for parameter in all_valid_params
            if parameter in values and np.isfinite(values.get(parameter, {}).get(SENSITIVITY_MEASURE, np.nan))
        }

    write_rank_table_csv(out_dir / f"scenario_parameter_ranks_{SENSITIVITY_MEASURE}.csv", rank_table, ordered_params, scenario_names)


def main() -> None:
    # Default: success metric (original behavior) -> write to base dir
    base_out = PROJECT_ROOT / "data" / "scenario_interaction_rank_analysis"
    # Load success scenario maps for aggregated CSV outputs later
    scenario_maps = load_scenario_scores(metric="success")
    _run_analysis_for_metric("success", base_out)

    # Also run for timesteps and write to a separate directory
    timesteps_out = PROJECT_ROOT / "data" / "scenario_interaction_rank_analysis_timesteps"
    _run_analysis_for_metric("timesteps", timesteps_out)

    # Also run for epistemicity and write to a separate directory
    epistemicity_out = PROJECT_ROOT / "data" / "scenario_interaction_rank_analysis_epistemicity"
    _run_analysis_for_metric("epistemicity", epistemicity_out)

    # Compute and write some aggregated per-scenario comparisons for the success metric
    output_dir = base_out
    mean_max_rows = compute_mean_max_spearman_by_scenario(scenario_maps)
    with (output_dir / "spearman_mean_vs_max_ranking_by_scenario.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scenario", "rho", "p_value", "n_params"])
        writer.writeheader()
        writer.writerows(mean_max_rows)

    sensitivity_max_rows = compute_sensitivity_max_spearman_by_scenario(scenario_maps)
    with (output_dir / "spearman_single_sensitivity_vs_max_interaction_by_scenario.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scenario", "rho", "p_value", "n_params"])
        writer.writeheader()
        writer.writerows(sensitivity_max_rows)

    print("\n=== Spearman: mean interaction ranking vs max interaction ranking ===")
    for row in mean_max_rows:
        print(row)
    print("\n=== Spearman: single sensitivity ranking vs max interaction ranking ===")
    for row in sensitivity_max_rows:
        print(row)

    combined_rows = []
    for measure in MEASURES:
        with (output_dir / f"spearman_pair_summary_{measure}.csv").open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                combined_rows.append(row)
    with (output_dir / "spearman_pair_summary_all_measures.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["measure", "scenario_a", "scenario_b", "rho", "p_value"])
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
