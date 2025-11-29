#!/usr/bin/env python3
"""
analyze_features.py

High-level analysis on features.db, focusing on:

1) Ecosystem-level stats:
   - Total repos
   - Repos with HEAD-level unstable features
   - Repos that ever used unstable features historically

2) Per-feature stats:
   - How many repos ever used a feature (history)
   - How many repos still use it on HEAD
   - How many repos retired it
   - Lifetime stats per feature (avg / median / min / max in days)

3) Category-level stats:
   - For each top_category:
       - total repos
       - repos with HEAD-level nightly usage
       - repos that ever used nightly historically

Outputs:
  - analysis_outputs/feature_head_summary.csv
  - analysis_outputs/feature_history_summary.csv
  - analysis_outputs/feature_lifetimes.csv
  - analysis_outputs/category_summary.csv
"""

import os
import sqlite3
import csv
from datetime import datetime
from statistics import mean, median

DB_PATH = "features.db"
OUT_DIR = "analysis_outputs"


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_iso(ts: str):
    """
    Parse an ISO8601 string like '2023-01-02T03:04:05Z' or with +00:00.
    Return a timezone-aware datetime in UTC, or None if invalid.
    """
    if ts is None:
        return None
    ts = ts.strip()
    if not ts:
        return None
    # Normalize 'Z' to '+00:00'
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def basic_overview(conn: sqlite3.Connection):
    log("=== BASIC OVERVIEW ===")
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM repos;")
    total_repos = cur.fetchone()[0] or 0

    try:
        cur.execute("SELECT COUNT(DISTINCT key) FROM repo_features;")
        head_repos = cur.fetchone()[0] or 0
    except sqlite3.OperationalError:
        head_repos = 0

    try:
        cur.execute("SELECT COUNT(DISTINCT key) FROM repo_feature_history;")
        ever_repos = cur.fetchone()[0] or 0
    except sqlite3.OperationalError:
        ever_repos = 0

    log(f"Total repos in sample: {total_repos}")
    log(f"Repos with HEAD-level unstable features (repo_features): {head_repos}")
    log(f"Repos that ever used unstable features (repo_feature_history): {ever_repos}")

    if total_repos > 0:
        log(f"HEAD-nightly ratio: {head_repos}/{total_repos} ≈ {head_repos/total_repos:.2%}")
        log(f"Ever-nightly ratio: {ever_repos}/{total_repos} ≈ {ever_repos/total_repos:.2%}")
    log("")


def feature_head_summary(conn: sqlite3.Connection, out_dir: str):
    """
    Dump per-feature HEAD-level usage summary into CSV.
    Columns: feature_name, head_repo_count
    """
    log("Generating feature_head_summary.csv ...")
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT feature_name, COUNT(*) AS repo_count
            FROM repo_features
            GROUP BY feature_name
            ORDER BY repo_count DESC, feature_name ASC;
            """
        )
    except sqlite3.OperationalError:
        log("repo_features table does not exist; skipping head summary.")
        return

    rows = cur.fetchall()
    out_path = os.path.join(out_dir, "feature_head_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name", "head_repo_count"])
        for feature_name, repo_count in rows:
            writer.writerow([feature_name, repo_count])

    log(f"Wrote {len(rows)} rows to {out_path}\n")


def feature_history_summary(conn: sqlite3.Connection, out_dir: str):
    """
    Dump per-feature history usage summary into CSV.
    Columns: feature_name, ever_repo_count
    """
    log("Generating feature_history_summary.csv ...")
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT feature_name, COUNT(*) AS repo_count
            FROM repo_feature_history
            GROUP BY feature_name
            ORDER BY repo_count DESC, feature_name ASC;
            """
        )
    except sqlite3.OperationalError:
        log("repo_feature_history table does not exist; skipping history summary.")
        return

    rows = cur.fetchall()
    out_path = os.path.join(out_dir, "feature_history_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name", "ever_repo_count"])
        for feature_name, repo_count in rows:
            writer.writerow([feature_name, repo_count])

    log(f"Wrote {len(rows)} rows to {out_path}\n")


def feature_lifetimes(conn: sqlite3.Connection, out_dir: str):
    """
    For each feature, compute lifetime stats across repos.

    Input: repo_feature_history rows:
      key, feature_name, first_seen_date, last_seen_date, still_present

    For each feature_name:
      - num_repos
      - num_still_present
      - num_retired
      - min_first_seen_date
      - max_first_seen_date
      - min_last_seen_date (non-null)
      - max_last_seen_date (non-null)
      - avg_lifetime_days (across repos where both dates exist)
      - median_lifetime_days
    """
    log("Computing feature_lifetimes.csv ...")
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT feature_name, key, first_seen_date, last_seen_date, still_present
            FROM repo_feature_history;
            """
        )
    except sqlite3.OperationalError:
        log("repo_feature_history table does not exist; skipping lifetimes.")
        return

    rows = cur.fetchall()
    if not rows:
        log("No rows in repo_feature_history; skipping lifetimes.")
        return

    per_feature = {}

    for feature_name, key, first_dt_str, last_dt_str, still_present in rows:
        d = per_feature.get(feature_name)
        if d is None:
            d = {
                "feature_name": feature_name,
                "num_repos": 0,
                "num_still_present": 0,
                "num_retired": 0,
                "first_dates": [],
                "last_dates": [],
                "lifetimes_days": [],
            }
            per_feature[feature_name] = d

        d["num_repos"] += 1
        if still_present:
            d["num_still_present"] += 1
        else:
            d["num_retired"] += 1

        first_dt = parse_iso(first_dt_str)
        last_dt = parse_iso(last_dt_str)

        if first_dt:
            d["first_dates"].append(first_dt)
        if last_dt:
            d["last_dates"].append(last_dt)
        if first_dt and last_dt:
            delta = (last_dt - first_dt).days
            # Only accept non-negative durations
            if delta >= 0:
                d["lifetimes_days"].append(delta)

    # Now aggregate per feature
    out_path = os.path.join(out_dir, "feature_lifetimes.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "feature_name",
            "num_repos",
            "num_still_present",
            "num_retired",
            "min_first_seen_date",
            "max_first_seen_date",
            "min_last_seen_date",
            "max_last_seen_date",
            "avg_lifetime_days",
            "median_lifetime_days",
        ])

        for feature_name in sorted(per_feature.keys()):
            d = per_feature[feature_name]
            first_dates = d["first_dates"]
            last_dates = d["last_dates"]
            lifetimes = d["lifetimes_days"]

            # Convert dates back to ISO strings for CSV
            min_first = min(first_dates).isoformat() if first_dates else ""
            max_first = max(first_dates).isoformat() if first_dates else ""
            min_last = min(last_dates).isoformat() if last_dates else ""
            max_last = max(last_dates).isoformat() if last_dates else ""

            avg_life = f"{mean(lifetimes):.2f}" if lifetimes else ""
            median_life = f"{median(lifetimes):.2f}" if lifetimes else ""

            writer.writerow([
                feature_name,
                d["num_repos"],
                d["num_still_present"],
                d["num_retired"],
                min_first,
                max_first,
                min_last,
                max_last,
                avg_life,
                median_life,
            ])

    log(f"Wrote feature lifetimes for {len(per_feature)} features to {out_path}\n")


def category_summary(conn: sqlite3.Connection, out_dir: str):
    """
    Summarize, per top_category:

      - total_repos_in_category
      - repos_with_head_nightly
      - repos_ever_nightly

    Note: repos.top_categories may contain multiple comma-separated entries.
    We treat each entry as a separate category membership.
    """
    log("Computing category_summary.csv ...")
    cur = conn.cursor()

    # Load repos & categories
    cur.execute("SELECT key, top_categories FROM repos;")
    repo_cats = cur.fetchall()

    # Load sets of repos with HEAD-level and EVER-level nightly usage
    try:
        cur.execute("SELECT DISTINCT key FROM repo_features;")
        head_keys = {row[0] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        head_keys = set()

    try:
        cur.execute("SELECT DISTINCT key FROM repo_feature_history;")
        ever_keys = {row[0] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        ever_keys = set()

    # Aggregate
    stats = {}  # category -> dict

    for key, cats_str in repo_cats:
        if not cats_str:
            cats = ["uncategorized"]
        else:
            # top_categories is stored as a comma-separated list in our pipeline
            cats = [c.strip() for c in cats_str.split(",") if c.strip()] or ["uncategorized"]

        for cat in cats:
            d = stats.get(cat)
            if d is None:
                d = {
                    "category": cat,
                    "total_repos": 0,
                    "head_nightly_repos": 0,
                    "ever_nightly_repos": 0,
                }
                stats[cat] = d

            d["total_repos"] += 1
            if key in head_keys:
                d["head_nightly_repos"] += 1
            if key in ever_keys:
                d["ever_nightly_repos"] += 1

    out_path = os.path.join(out_dir, "category_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category",
            "total_repos",
            "head_nightly_repos",
            "ever_nightly_repos",
            "head_nightly_ratio",
            "ever_nightly_ratio",
        ])

        for cat, d in sorted(stats.items(), key=lambda kv: kv[1]["total_repos"], reverse=True):
            total = d["total_repos"]
            head_nightly = d["head_nightly_repos"]
            ever_nightly = d["ever_nightly_repos"]
            head_ratio = head_nightly / total if total > 0 else 0.0
            ever_ratio = ever_nightly / total if total > 0 else 0.0

            writer.writerow([
                cat,
                total,
                head_nightly,
                ever_nightly,
                f"{head_ratio:.4f}",
                f"{ever_ratio:.4f}",
            ])

    log(f"Wrote category summary for {len(stats)} categories to {out_path}\n")


def main():
    ensure_out_dir(OUT_DIR)

    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found. Run scan_features.py and scan_history.py first.")
        return

    conn = sqlite3.connect(DB_PATH)

    try:
        basic_overview(conn)
        feature_head_summary(conn, OUT_DIR)
        feature_history_summary(conn, OUT_DIR)
        feature_lifetimes(conn, OUT_DIR)
        category_summary(conn, OUT_DIR)
    finally:
        conn.close()

    log("All analysis done.")


if __name__ == "__main__":
    main()
