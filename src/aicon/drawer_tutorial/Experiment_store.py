import sqlite3
import hashlib
import json
import time
import os
from typing import Dict, Any, List, Optional
import numpy as np


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

            # Trial table: one row per simulation run
            cur.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_id TEXT,
                seed INTEGER,
                success INTEGER,
                timesteps INTEGER,
                error REAL,
                metadata TEXT,
                FOREIGN KEY(config_id) REFERENCES configs(config_id)
            )
            """)

            # If the table already exists but metadata is missing, migrate it.
            cur.execute("PRAGMA table_info(trials)")
            columns = [row[1] for row in cur.fetchall()]
            if "metadata" not in columns:
                cur.execute("ALTER TABLE trials ADD COLUMN metadata TEXT")

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

    def add_trial(
        self,
        params: Dict[str, Any],
        success: bool,
        seed: int,
        timesteps: int,
        error: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        config_id = _hash_config(params)
        metadata_json = None if metadata is None else json.dumps(metadata)

        def _insert_trial():
            cur = self.conn.cursor()
            cur.execute("""
            INSERT OR IGNORE INTO configs (config_id, params)
            VALUES (?, ?)
            """, (config_id, json.dumps(params)))
            cur.execute("""
            INSERT INTO trials (config_id, seed, success, timesteps, error, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                config_id,
                seed,
                int(success),
                timesteps,
                None if error is None else float(error),
                metadata_json,
            ))
            self.conn.commit()

        self._run_with_lock_retry(_insert_trial)

    # ----------------------------
    # Query trials for config
    # ----------------------------

    def get_trials(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        config_id = _hash_config(params)

        cur = self.conn.cursor()
        cur.execute("""
        SELECT success, timesteps, error, metadata
        FROM trials
        WHERE config_id = ?
        """, (config_id,))

        rows = cur.fetchall()
        return [
            {
                "success": r[0],
                "timesteps": r[1],
                "error": r[2],
                "metadata": json.loads(r[3]) if r[3] is not None else None,
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