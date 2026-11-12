"""
Unified database merge script for sweep experiments.

Merges per-job shard databases (created by both Sweep_hpc.py and Pairwise_Sweep.py)
into a single analysis database.

Usage:
    python merge_sweep_databases.py <data_directory> [--output <output_db>]
    
    data_directory: Directory containing experiment_store_job_*.db files
    --output: Path to target merged database (default: data_directory/experiment_store.db)
"""

import sys
import json
import sqlite3
from pathlib import Path

from aicon.drawer_tutorial.Sweep_files.Experiment_store import ExperimentStore


DB_SHARD_PREFIX = "experiment_store_job_"


def merge_sharded_databases(source_dir: Path, target_db: Path) -> None:
    """Merge per-job shard databases into a single analysis database.
    
    Args:
        source_dir: Directory containing experiment_store_job_*.db shard files
        target_db: Path to output merged database
    """
    shard_paths = sorted(source_dir.glob(f"{DB_SHARD_PREFIX}*.db"))
    if not shard_paths:
        print(f"[MERGE] No shard databases found in {source_dir}")
        return

    target_db.parent.mkdir(parents=True, exist_ok=True)
    target_store = ExperimentStore(str(target_db))

    try:
        for shard_path in shard_paths:
            if shard_path.resolve() == target_db.resolve():
                continue

            print(f"[MERGE] Merging {shard_path.name} into {target_db.name}")
            try:
                target_store.conn.execute("ATTACH DATABASE ? AS srcdb", (str(shard_path),))
                target_store.conn.execute("BEGIN")

                # Merge configs
                target_store.conn.execute(
                    "INSERT OR IGNORE INTO configs (config_id, params) SELECT config_id, params FROM srcdb.configs"
                )

                # Merge trace_catalog: ensure every src trace_hash exists in target and build mapping
                src_trace_rows = target_store.conn.execute(
                    "SELECT name FROM srcdb.sqlite_master WHERE type='table' AND name='trace_catalog'"
                ).fetchone()
                src_to_target_trace = {}
                if src_trace_rows is not None:
                    rows = target_store.conn.execute(
                        "SELECT trace_id, trace_hash, trace_text FROM srcdb.trace_catalog"
                    ).fetchall()
                    for src_id, trace_hash, trace_text in rows:
                        cur = target_store.conn.execute(
                            "SELECT trace_id FROM trace_catalog WHERE trace_hash = ?",
                            (trace_hash,)
                        ).fetchone()
                        if cur is not None:
                            target_id = int(cur[0])
                        else:
                            target_store.conn.execute(
                                "INSERT INTO trace_catalog (trace_hash, trace_text) VALUES (?, ?)",
                                (trace_hash, trace_text),
                            )
                            target_id = int(target_store.conn.execute("SELECT last_insert_rowid()").fetchone()[0])
                        src_to_target_trace[int(src_id)] = target_id

                # Merge trials, remapping trace_ids from src -> target using src_to_target_trace
                src_trials = target_store.conn.execute(
                    "SELECT config_id, seed, success, timesteps, error, grasp, actual_grasped, kinematic_axis_error, anchor_error, true_joint, metadata, trace_ids FROM srcdb.trials"
                ).fetchall()

                for row in src_trials:
                    (config_id, seed, success, timesteps, error, grasp, actual_grasped, kinematic_axis_error, anchor_error, true_joint, metadata, trace_ids_json) = row

                    # remap trace ids if present
                    trace_ids_target_json = None
                    if trace_ids_json:
                        try:
                            src_trace_ids = json.loads(trace_ids_json)
                        except Exception:
                            src_trace_ids = []
                        mapped = []
                        for sid in src_trace_ids:
                            mapped_id = src_to_target_trace.get(int(sid))
                            if mapped_id is None:
                                # As a fallback, try to resolve by hash from srcdb.trace_catalog
                                cur = target_store.conn.execute(
                                    "SELECT trace_hash FROM srcdb.trace_catalog WHERE trace_id = ?",
                                    (int(sid),),
                                ).fetchone()
                                if cur is not None:
                                    th = cur[0]
                                    existing = target_store.conn.execute(
                                        "SELECT trace_id FROM trace_catalog WHERE trace_hash = ?",
                                        (th,),
                                    ).fetchone()
                                    if existing is not None:
                                        mapped_id = int(existing[0])
                                    else:
                                        # copy the trace_text over
                                        trace_text_row = target_store.conn.execute(
                                            "SELECT trace_text FROM srcdb.trace_catalog WHERE trace_id = ?",
                                            (int(sid),),
                                        ).fetchone()
                                        trace_text = trace_text_row[0] if trace_text_row is not None else ""
                                        target_store.conn.execute(
                                            "INSERT INTO trace_catalog (trace_hash, trace_text) VALUES (?, ?)",
                                            (th, trace_text),
                                        )
                                        mapped_id = int(target_store.conn.execute("SELECT last_insert_rowid()").fetchone()[0])
                            if mapped_id is not None:
                                mapped.append(mapped_id)
                        trace_ids_target_json = None if not mapped else json.dumps(mapped)

                    # insert trial with trace_ids included, or update an existing row from a previous shard
                    columns = [
                        "config_id",
                        "seed",
                        "success",
                        "timesteps",
                        "error",
                    ]
                    values = [config_id, seed, success, timesteps, error]

                    if "grasp" in target_store.conn.execute("PRAGMA table_info(trials)").fetchall()[0] and False:
                        pass

                    trial_columns = {row[1] for row in target_store.conn.execute("PRAGMA table_info(trials)").fetchall()}
                    if "grasp" in trial_columns:
                        columns.append("grasp")
                        values.append(grasp)
                    if "actual_grasped" in trial_columns:
                        columns.append("actual_grasped")
                        values.append(actual_grasped)
                    if "kinematic_axis_error" in trial_columns:
                        columns.append("kinematic_axis_error")
                        values.append(kinematic_axis_error)
                    if "anchor_error" in trial_columns:
                        columns.append("anchor_error")
                        values.append(anchor_error)
                    if "true_joint" in trial_columns:
                        columns.append("true_joint")
                        values.append(true_joint)
                    if "trace_ids" in trial_columns:
                        columns.append("trace_ids")
                        values.append(trace_ids_target_json)
                    if "metadata" in trial_columns:
                        columns.append("metadata")
                        values.append(metadata)

                    placeholders = ", ".join(["?"] * len(columns))
                    update_clause = ", ".join(
                        f"{column}=excluded.{column}"
                        for column in columns
                        if column not in ("config_id", "seed")
                    )
                    target_store.conn.execute(
                        f"INSERT INTO trials ({', '.join(columns)}) VALUES ({placeholders}) ON CONFLICT(config_id, seed) DO UPDATE SET {update_clause}",
                        values,
                    )

                target_store.conn.commit()
            finally:
                try:
                    target_store.conn.execute("DETACH DATABASE srcdb")
                except Exception:
                    pass

        print(f"[MERGE] Merge complete: {target_db}")
    finally:
        target_store.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    source_dir = Path(sys.argv[1])
    if not source_dir.exists():
        print(f"Error: Directory {source_dir} does not exist")
        sys.exit(1)

    target_db = None
    if len(sys.argv) > 2 and sys.argv[2] == "--output":
        if len(sys.argv) > 3:
            target_db = Path(sys.argv[3])
        else:
            print("Error: --output requires a path argument")
            sys.exit(1)
    
    if target_db is None:
        target_db = source_dir / "experiment_store.db"

    merge_sharded_databases(source_dir, target_db)
