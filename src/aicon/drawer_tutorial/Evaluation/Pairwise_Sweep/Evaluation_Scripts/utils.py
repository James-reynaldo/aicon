"""Utility functions for pairwise sweep analysis."""
from collections import defaultdict
from pathlib import Path
import json
import sqlite3
import numpy as np
import os

SI_SEPERATE = False

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _get_matplotlib_pyplot():
    """Get matplotlib pyplot, with error handling."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting functions. Install it or run with --debug-success-rates to inspect rates without plotting."
        ) from exc
    return plt


ONE_D_LABELS_TO_INCLUDE = {"0.2x", "0.5x", "2x", "5x", "standard"}


def load_sweep_plan():
    """Load sweep plan from JSON to map value indices to multiplier labels."""
    # Try multiple possible paths to locate the JSON file
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "Sweep_files" / "sweep_plan_actual_values.json",
        Path(__file__).parent.parent.parent.parent.parent / "Sweep_files" / "sweep_plan_actual_values.json",
        Path("/scratch/aldo/aicon/src/aicon/drawer_tutorial/Sweep_files/sweep_plan_actual_values.json"),
    ]
    
    for json_path in possible_paths:
        try:
            with open(json_path) as f:
                data = json.load(f)
            # Build a mapping from multiplier to label
            multiplier_to_label = {}
            sweep_multipliers = data.get("sweep_multipliers", [])
            for idx, multiplier in enumerate(sweep_multipliers):
                if multiplier == -1:
                    label = "standard"
                elif multiplier == 0:
                    label = "0"
                else:
                    label = f"{multiplier}x"
                multiplier_to_label[idx] = label
            return multiplier_to_label
        except Exception:
            continue
    
    print(f"[WARNING] Could not load sweep plan from any path")
    return {}


SWEEP_PLAN_MULTIPLIERS = load_sweep_plan()


def value_index_to_label(value_str):
    """Convert 'value_X' format to actual multiplier label."""
    try:
        if value_str.startswith("value_"):
            idx = int(value_str.split("_")[1])
            label = SWEEP_PLAN_MULTIPLIERS.get(idx, value_str)
            return label
    except Exception:
        pass
    return value_str


def short_label(label):
    """Extract short label from full parameter name."""
    # Special case: connection.DistGraspHandConnection.close_dist_threshold -> "close dist threshold"
    if label == "connection.DistGraspHandConnection.close_dist_threshold":
        return "close dist threshold"
    return label.split(".")[-1]


def normalize_one_d_label(label):
    """Normalize one-dimensional sweep labels."""
    if label is None:
        return ""
    label = str(label).strip()
    if label == "standard":
        return "standard"
    if label.startswith("x"):
        normalized = label[1:]
        if normalized in {"0.2", "0.5", "2", "5"}:
            return f"{normalized}x"
    if label.endswith("x"):
        normalized = label[:-1]
        if normalized in {"0.2", "0.5", "2", "5"}:
            return label
    return label


def parse_pair(metadata):
    """Parse metadata to extract parameter pair."""
    if not metadata:
        return None

    raw_parameters = metadata.get("parameters")
    if not raw_parameters:
        return None

    sweep_labels = metadata.get("sweep_labels", "")
    pair_size = len([part for part in str(sweep_labels).split(",") if part.strip()])
    if pair_size <= 0:
        pair_size = 2

    if "." in raw_parameters:
        group_part, param_part = raw_parameters.rsplit(".", 1)
        groups = [part.strip() for part in group_part.split(",")]
        params = [part.strip() for part in param_part.split(",")]
        if len(params) == pair_size:
            if len(groups) == len(params):
                return tuple(f"{group}.{param}" for group, param in zip(groups, params))
            return tuple(params)

    params = [part.strip() for part in raw_parameters.split(",")]
    if len(params) == pair_size:
        return tuple(params)

    return None


def _filter_rows_by_trial_window(rows, trial_window="all"):
    """Restrict each run to all / first-5 / last-5 trials by seed ordering."""
    if trial_window == "all":
        return rows

    grouped = defaultdict(list)
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) < 2:
            continue
        config_id = row[0]
        seed = row[1]
        try:
            seed_value = int(seed)
        except (TypeError, ValueError):
            continue
        grouped[str(config_id)].append((seed_value, row))

    filtered = []
    for entries in grouped.values():
        entries.sort(key=lambda item: item[0])
        if trial_window == "first5":
            selected = entries[:5]
        elif trial_window == "last5":
            selected = entries[-5:]
        else:
            selected = entries
        filtered.extend(row for _, row in selected)
    return filtered


def resolve_parameter_selection(selection, parameters):
    """Resolve user-provided parameter names against full names and short labels."""
    if selection is None:
        return []

    resolved = []
    seen = set()
    for item in selection:
        candidate = str(item).strip()
        if not candidate:
            continue

        matches = [
            parameter
            for parameter in parameters
            if parameter == candidate
            or short_label(parameter) == candidate
            or parameter.endswith(f".{candidate}")
        ]
        for parameter in matches:
            if parameter not in seen:
                resolved.append(parameter)
                seen.add(parameter)

    return resolved
