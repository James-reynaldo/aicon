import sqlite3
import hashlib
import json
import time
import os
from typing import Dict, Any, List, Optional
import numpy as np


def _canonicalize_trace(trace: Any) -> Optional[str]:
    """Convert a trace into a canonical JSON string for deterministic hashing."""
    if trace is None:
        return None
    if isinstance(trace, np.ndarray):
        data = trace.tolist()
    elif isinstance(trace, (list, tuple)):
        data = list(trace)
    elif isinstance(trace, dict):
        data = {str(k): _canonicalize_trace(v) if isinstance(v, (list, tuple, dict, np.ndarray)) else v for k, v in trace.items()}
    elif isinstance(trace, str):
        data = trace
    else:
        try:
            data = json.loads(json.dumps(trace))
        except TypeError:
            data = str(trace)
    try:
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError:
        return json.dumps(str(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _trace_hash(trace: Any) -> Optional[str]:
    canonical = _canonicalize_trace(trace)
    if canonical is None:
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stable_trace_id(trace_hash: str) -> int:
    """Return a deterministic integer id derived from the trace hash.

    This keeps trace identity stable across shards and merged databases instead of
    depending on each SQLite database's local AUTOINCREMENT sequence.
    """
    # Use the first 8 bytes as a 64-bit positive integer, with a small collision
    # guard if the value is already taken in the catalog.
    digest = hashlib.sha256(trace_hash.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


# ----------------------------
# Utilities
# ----------------------------

def _hash_config(params: Dict[str, Any]) -> str:
    """
    Create a stable hash for a parameter configuration.
    Ensures identical configs map to same config_id.
    """
    encoded = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


# ----------------------------
# Database Layer
# ----------------------------

class ExperimentStore:
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self.conn = self._open_connection()
        self._create_tables()

    def _open_connection(self):
        for attempt in range(6):
            conn = sqlite3.connect(self.db_path, timeout=120.0)
            conn.execute("PRAGMA busy_timeout = 120000")

            # WAL is often fragile on network/shared filesystems used by HPC jobs.
            # Setting journal mode is optional here: if another process currently
            # holds a lock, proceed with the existing mode instead of failing startup.
            journal_mode = "WAL" if os.getenv("AICON_SQLITE_WAL", "0") == "1" else "DELETE"
            try:
                conn.execute(f"PRAGMA journal_mode = {journal_mode}")
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 5:
                    # Non-lock failures should still propagate.
                    if "locked" not in str(exc).lower():
                        conn.close()
                        raise
                conn.close()
                time.sleep(min(0.25 * (2 ** attempt), 3.0))
                continue

            return conn

        # Fallback: open connection without touching journal mode.
        conn = sqlite3.connect(self.db_path, timeout=120.0)
        conn.execute("PRAGMA busy_timeout = 120000")
        return conn

    def _reconnect(self):
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = self._open_connection()

    def _run_with_lock_retry(self, fn, *, max_retries: int = 8):
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                is_locked = "locked" in msg
                is_disk_io = "disk i/o" in msg

                if (not is_locked and not is_disk_io) or attempt == max_retries:
                    raise

                try:
                    self.conn.rollback()
                except sqlite3.Error:
                    pass

                if is_disk_io:
                    self._reconnect()

                time.sleep(min(0.25 * (2 ** attempt), 5.0))

    def _create_tables(self):
        def _create():
            cur = self.conn.cursor()

            # Config table: one row per parameter setting
            cur.execute("""
            CREATE TABLE IF NOT EXISTS configs (
                config_id TEXT PRIMARY KEY,
                params TEXT
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS trace_catalog (
                trace_id INTEGER PRIMARY KEY,
                trace_hash TEXT UNIQUE NOT NULL,
                trace_text TEXT NOT NULL
            )
            """)

            # Trial table: one row per simulation run
            cur.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_id TEXT,
                seed INTEGER,
                success INTEGER,
                timesteps INTEGER,
                error REAL,
                true_joint REAL,
                grasp INTEGER,
                actual_grasped INTEGER,
                kinematic_axis_error REAL,
                anchor_error REAL,
                trace_ids TEXT,
                metadata TEXT,
                FOREIGN KEY(config_id) REFERENCES configs(config_id)
            )
            """)

            # Migrate older schemas that predate metadata and the newer error columns.
            cur.execute("PRAGMA table_info(trials)")
            columns = [row[1] for row in cur.fetchall()]
            if "metadata" not in columns:
                cur.execute("ALTER TABLE trials ADD COLUMN metadata TEXT")
                columns = [row[1] for row in self.conn.execute("PRAGMA table_info(trials)").fetchall()]
            if "trace_ids" not in columns:
                cur.execute("ALTER TABLE trials ADD COLUMN trace_ids TEXT")
                columns = [row[1] for row in self.conn.execute("PRAGMA table_info(trials)").fetchall()]
            for column_name in ("true_joint", "grasp", "actual_grasped", "kinematic_axis_error", "anchor_error"):
                if column_name not in columns:
                    column_type = "INTEGER" if column_name in ("grasp", "actual_grasped") else "REAL"
                    cur.execute(f"ALTER TABLE trials ADD COLUMN {column_name} {column_type}")
                    columns = [row[1] for row in self.conn.execute("PRAGMA table_info(trials)").fetchall()]
            self.conn.commit()

        self._run_with_lock_retry(_create)

    # ----------------------------
    # Insert config
    # ----------------------------

    def add_config(self, params: Dict[str, Any]) -> str:
        config_id = _hash_config(params)

        def _insert_config():
            cur = self.conn.cursor()
            cur.execute("""
            INSERT OR IGNORE INTO configs (config_id, params)
            VALUES (?, ?)
            """, (config_id, json.dumps(params)))
            self.conn.commit()

        self._run_with_lock_retry(_insert_config)
        return config_id

    # ----------------------------
    # Insert trial
    # ----------------------------

    def register_trace(self, trace: Any) -> int:
        """Return a compact integer trace_id for a canonical trace.

        The integer id is derived deterministically from the trace hash so that the
        same trace receives the same index across separate databases and after shard
        merges, instead of depending on SQLite AUTOINCREMENT state.
        """
        canonical = _canonicalize_trace(trace)
        trace_hash = _trace_hash(trace)
        if canonical is None or trace_hash is None:
            raise ValueError("trace must be non-empty and serializable")

        def _register():
            cur = self.conn.cursor()
            cur.execute(
                "SELECT trace_id FROM trace_catalog WHERE trace_hash = ?",
                (trace_hash,),
            )
            row = cur.fetchone()
            if row is not None:
                return int(row[0])

            trace_id = _stable_trace_id(trace_hash)
            while True:
                cur.execute(
                    "SELECT 1 FROM trace_catalog WHERE trace_id = ?",
                    (trace_id,),
                )
                if cur.fetchone() is None:
                    break
                trace_id = (trace_id + 1) & ((1 << 63) - 1)

            cur.execute(
                "INSERT INTO trace_catalog (trace_id, trace_hash, trace_text) VALUES (?, ?, ?)",
                (trace_id, trace_hash, canonical),
            )
            self.conn.commit()
            return int(trace_id)

        return self._run_with_lock_retry(_register)

    def get_trace_by_id(self, trace_id: int) -> Optional[Dict[str, Any]]:
        """Return the catalog entry for a given trace_id or None if missing.

        Returns a dict with keys: `trace_id`, `trace_hash`, `trace_text`.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT trace_hash, trace_text FROM trace_catalog WHERE trace_id = ?", (trace_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return {"trace_id": int(trace_id), "trace_hash": row[0], "trace_text": row[1]}

    def get_trace_id_by_hash(self, trace_hash: str) -> Optional[int]:
        """Return the compact trace_id for a given SHA-256 trace_hash, or None if not present."""
        cur = self.conn.cursor()
        cur.execute("SELECT trace_id FROM trace_catalog WHERE trace_hash = ?", (trace_hash,))
        row = cur.fetchone()
        return int(row[0]) if row is not None else None

    def resolve_trace_ids(self, trace_ids: Optional[List[int]]) -> List[Optional[str]]:
        """Resolve a list of integer trace_ids into their canonical trace_text strings.

        Returns a list with the same length as `trace_ids`. Missing ids map to None.
        """
        if not trace_ids:
            return []
        cur = self.conn.cursor()
        results = []
        for tid in trace_ids:
            cur.execute("SELECT trace_text FROM trace_catalog WHERE trace_id = ?", (tid,))
            row = cur.fetchone()
            results.append(row[0] if row is not None else None)
        return results

    def add_trial(
        self,
        params: Dict[str, Any],
        success: bool,
        seed: int,
        timesteps: int,
        error: Optional[float] = None,
        grasp: Optional[bool] = None,
        actual_grasped: Optional[bool] = None,
        kinematic_axis_error: Optional[float] = None,
        anchor_error: Optional[float] = None,
        true_joint: Optional[float] = None,
        trace_ids: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        config_id = _hash_config(params)
        metadata_json = None if metadata is None else json.dumps(metadata)
        trace_ids_json = None if trace_ids is None else json.dumps(trace_ids)

        def _insert_trial():
            cur = self.conn.cursor()
            cur.execute("""
            INSERT OR IGNORE INTO configs (config_id, params)
            VALUES (?, ?)
            """, (config_id, json.dumps(params)))
            cur.execute("PRAGMA table_info(trials)")
            trial_columns = {row[1] for row in cur.fetchall()}
            insert_columns = [
                "config_id",
                "seed",
                "success",
                "timesteps",
                "error",
            ]
            insert_values = [
                config_id,
                seed,
                int(success),
                timesteps,
                None if error is None else float(error),
            ]

            if "grasp" in trial_columns:
                insert_columns.append("grasp")
                insert_values.append(None if grasp is None else int(grasp))
            if "actual_grasped" in trial_columns:
                insert_columns.append("actual_grasped")
                insert_values.append(None if actual_grasped is None else int(actual_grasped))
            if "kinematic_axis_error" in trial_columns:
                insert_columns.append("kinematic_axis_error")
                insert_values.append(None if kinematic_axis_error is None else float(kinematic_axis_error))
            if "anchor_error" in trial_columns:
                insert_columns.append("anchor_error")
                insert_values.append(None if anchor_error is None else float(anchor_error))
            if "true_joint" in trial_columns:
                insert_columns.append("true_joint")
                insert_values.append(None if true_joint is None else float(true_joint))
            if "trace_ids" in trial_columns:
                insert_columns.append("trace_ids")
                insert_values.append(trace_ids_json)
            if "metadata" in trial_columns:
                insert_columns.append("metadata")
                insert_values.append(metadata_json)

            placeholders = ", ".join(["?"] * len(insert_columns))
            cur.execute(
                f"INSERT INTO trials ({', '.join(insert_columns)}) VALUES ({placeholders})",
                insert_values,
            )
            self.conn.commit()

        self._run_with_lock_retry(_insert_trial)

    # ----------------------------
    # Query trials for config
    # ----------------------------

    def get_trials(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        config_id = _hash_config(params)

        cur = self.conn.cursor()
        cur.execute("""
        SELECT success, timesteps, error, grasp, kinematic_axis_error, anchor_error, true_joint, metadata
        FROM trials
        WHERE config_id = ?
        """, (config_id,))

        rows = cur.fetchall()
        return [
            {
                "success": r[0],
                "timesteps": r[1],
                "error": r[2],
                "metadata": json.loads(r[7]) if r[7] is not None else None,
                "grasp": r[3],
                "kinematic_axis_error": r[4],
                "anchor_error": r[5],
                "true_joint": r[6]
            }
            for r in rows
        ]

    # ----------------------------
    # Aggregation (VERY important for GSA)
    # ----------------------------

    def get_summary(self, params: Dict[str, Any]) -> Dict[str, float]:
        trials = self.get_trials(params)

        if len(trials) == 0:
            return {
                "n": 0,
                "success_rate": None,
                "std": None
            }

        success = np.array([t["success"] for t in trials])

        return {
            "n": len(success),
            "success_rate": float(np.mean(success)),
            "std": float(np.std(success))
        }

    # ----------------------------
    # Batch export (for ML / Sobol / surrogate)
    # ----------------------------

    def export_dataset(self):
        cur = self.conn.cursor()

        cur.execute("""
        SELECT c.params, t.success
        FROM trials t
        JOIN configs c ON t.config_id = c.config_id
        """)

        data = cur.fetchall()

        X = []
        y = []

        for params_json, success in data:
            params = json.loads(params_json)
            X.append(params)
            y.append(success)

        return X, np.array(y)

    # ----------------------------
    # Close
    # ----------------------------

    def close(self):
        self.conn.close()