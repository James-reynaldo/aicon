#!/usr/bin/env python3

import argparse
import sqlite3


def remove_duplicates(db_path):
    conn = sqlite3.connect(db_path)

    try:
        cursor = conn.cursor()

        before = conn.total_changes

        cursor.execute("""
            WITH duplicates AS (
                SELECT trial_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY config_id, seed
                           ORDER BY trial_id
                       ) AS rn
                FROM trials
            )
            DELETE FROM trials
            WHERE trial_id IN (
                SELECT trial_id
                FROM duplicates
                WHERE rn > 1
            );
        """)

        conn.commit()

        deleted = conn.total_changes - before

        print(f"Removed {deleted} duplicate trial(s) from {db_path}")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate trials based on (config_id, seed)."
    )

    parser.add_argument(
        "database",
        help="Path to the SQLite database"
    )

    args = parser.parse_args()

    remove_duplicates(args.database)


if __name__ == "__main__":
    main()