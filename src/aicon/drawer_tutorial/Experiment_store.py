import sqlite3
import hashlib
import json
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
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
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

    # ----------------------------
    # Insert config
    # ----------------------------

    def add_config(self, params: Dict[str, Any]) -> str:
        config_id = _hash_config(params)

        cur = self.conn.cursor()
        cur.execute("""
        INSERT OR IGNORE INTO configs (config_id, params)
        VALUES (?, ?)
        """, (config_id, json.dumps(params)))

        self.conn.commit()
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
        config_id = self.add_config(params)

        metadata_json = None if metadata is None else json.dumps(metadata)

        cur = self.conn.cursor()
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