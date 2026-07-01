import argparse
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


DEFAULT_LABELS = {"x0.2", "0.2x", "x5", "5x"}
STANDARD_LABELS = {"standard", "1x", "x1"}


def parse_labels(metadata_json):
    if not metadata_json:
        return None

    try:
        metadata = json.loads(metadata_json)
    except (TypeError, json.JSONDecodeError):
        return None

    if metadata.get("experiment_type") != "pairwise_sweep":
        return None

    labels = [
        label.strip()
        for label in str(metadata.get("sweep_labels", "")).split(",")
        if label.strip()
    ]
    if len(labels) != 2:
        return None

    return labels


def should_delete(labels, low_labels):
    first, second = labels
    return (
        first in low_labels
        and second in STANDARD_LABELS
        or second in low_labels
        and first in STANDARD_LABELS
    )


def find_trials_to_delete(conn, low_labels):
    rows = conn.execute(
        """
        SELECT trial_id, config_id, metadata
        FROM trials
        WHERE metadata IS NOT NULL
        """
    ).fetchall()

    matched = []
    skipped_pairwise = 0
    for trial_id, config_id, metadata_json in rows:
        labels = parse_labels(metadata_json)
        if labels is None:
            continue
        if should_delete(labels, low_labels):
            matched.append((trial_id, config_id, ",".join(labels)))
        else:
            skipped_pairwise += 1

    return matched, skipped_pairwise


def backup_db(db_path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.backup-{timestamp}")
    shutil.copy2(db_path, backup_path)
    return backup_path


def delete_trials(db_path, low_labels, execute=False, create_backup=True):
    conn = sqlite3.connect(db_path)
    try:
        matched, skipped_pairwise = find_trials_to_delete(conn, low_labels)
        trial_ids = [trial_id for trial_id, _, _ in matched]
        config_ids = sorted({config_id for _, config_id, _ in matched})

        print(f"Database: {db_path}")
        print(f"Matched trials to delete: {len(trial_ids)}")
        print(f"Matched configs touched: {len(config_ids)}")
        print(f"Other pairwise trials left alone: {skipped_pairwise}")

        if matched:
            label_counts = {}
            for _, _, labels in matched:
                label_counts[labels] = label_counts.get(labels, 0) + 1
            print("Matched label combinations:")
            for labels, count in sorted(label_counts.items()):
                print(f"  {labels}: {count}")

        if not execute:
            print("\nDry run only. Re-run with --execute to delete these trials.")
            return

        backup_path = None
        if create_backup:
            backup_path = backup_db(db_path)
            print(f"Backup written to: {backup_path}")

        conn.execute("BEGIN")
        conn.executemany(
            "DELETE FROM trials WHERE trial_id = ?",
            [(trial_id,) for trial_id in trial_ids],
        )
        orphaned_configs = conn.execute(
            """
            SELECT c.config_id
            FROM configs c
            LEFT JOIN trials t ON t.config_id = c.config_id
            WHERE t.config_id IS NULL
            """
        ).fetchall()
        conn.executemany(
            "DELETE FROM configs WHERE config_id = ?",
            orphaned_configs,
        )
        conn.commit()

        print(f"Deleted trials: {len(trial_ids)}")
        print(f"Deleted orphaned configs: {len(orphaned_configs)}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Delete pairwise_sweep trials where one label is standard and the "
            "other is a low 1D label, e.g. x0.2,standard or standard,x0.5."
        )
    )
    parser.add_argument("db_path", type=Path, help="Path to experiment_store.db")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=sorted(DEFAULT_LABELS),
        help="Low sweep labels to delete when paired with standard.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete rows. Without this, only prints what would be deleted.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped database backup before deleting.",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        raise SystemExit(f"Database not found: {args.db_path}")

    delete_trials(
        args.db_path,
        low_labels=set(args.labels),
        execute=args.execute,
        create_backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
