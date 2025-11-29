#!/usr/bin/env python3
"""
scan_history.py

Reconstruct historical usage of Rust unstable features (#![feature(..)])
from git history.

This script assumes:
- features.db exists and has at least the `repos` table
  (created by scan_features.py).
- Git repositories have already been cloned under the paths recorded
  in `repos.path`.

It will:
- For each repository:
  - Run `git log -1` to get HEAD commit & timestamp.
  - Run `git log --reverse -p -S 'feature(' -- '*.rs'` to find commits
    where the number of occurrences of 'feature(' changed.
  - Parse added/removed inner attributes to infer when each feature
    gate was first introduced and (possibly) removed.
- Store per-(repo, feature) history in SQLite table `repo_feature_history`.

Usage examples:

  # Scan all repos recorded in features.db
  python3 scan_history.py \
    --db features.db \
    --workers 4

  # Only scan repos that currently use unstable features on HEAD
  python3 scan_history.py \
    --db features.db \
    --only-head-nightly \
    --workers 4

  # Limit to first 50 repos (for testing)
  python3 scan_history.py \
    --db features.db \
    --workers 4 \
    --max-repos 50
"""

import argparse
import os
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


# ========================= Logging helpers =========================

def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


# ========================= Data structures =========================

@dataclass
class RepoHistoryJob:
    key: str       # "owner/repo"
    path: str      # local filesystem path to repo
    owner: str
    name: str


@dataclass
class HistoryResult:
    key: str
    success: bool
    error: str
    rows: List[Tuple[str, str, str, str, Optional[str], Optional[str], int]]
    # row = (key, feature_name, first_commit, first_date, last_commit, last_date, still_present)


# ========================= SQLite helpers =========================

def init_db(db_path: str) -> sqlite3.Connection:
    """
    Open SQLite connection and make sure repo_feature_history exists.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS repo_feature_history (
          key              TEXT,
          feature_name     TEXT,
          first_seen_commit TEXT,
          first_seen_date   TEXT,
          last_seen_commit  TEXT,
          last_seen_date    TEXT,
          still_present     INTEGER,
          PRIMARY KEY (key, feature_name),
          FOREIGN KEY (key) REFERENCES repos(key)
        );
        """
    )
    conn.commit()
    return conn


def load_repos(conn: sqlite3.Connection) -> Dict[str, RepoHistoryJob]:
    """
    Load all repos from the `repos` table.
    """
    cur = conn.cursor()
    cur.execute("SELECT key, owner, name, path FROM repos;")
    jobs: Dict[str, RepoHistoryJob] = {}
    for key, owner, name, path in cur.fetchall():
        if not path:
            continue
        jobs[key] = RepoHistoryJob(key=key, path=path, owner=owner or "", name=name or "")
    return jobs


def load_head_features(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    """
    Load HEAD-level feature usage from repo_features table:
    key -> list of feature_name that are present on HEAD.
    """
    cur = conn.cursor()
    try:
        cur.execute("SELECT key, feature_name FROM repo_features;")
    except sqlite3.OperationalError:
        # repo_features may not exist yet if scan_features.py hasn't been run.
        return {}

    head_features: Dict[str, List[str]] = defaultdict(list)
    for key, feature_name in cur.fetchall():
        head_features[key].append(feature_name)
    return head_features


def replace_repo_history(
    conn: sqlite3.Connection,
    key: str,
    rows: List[Tuple[str, str, str, str, Optional[str], Optional[str], int]],
) -> None:
    """
    Replace history rows for a given repo:
      - DELETE existing rows for this key
      - INSERT new rows (if any)
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM repo_feature_history WHERE key = ?;", (key,))
    if rows:
        cur.executemany(
            """
            INSERT INTO repo_feature_history
              (key, feature_name,
               first_seen_commit, first_seen_date,
               last_seen_commit,  last_seen_date,
               still_present)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
    conn.commit()


# ========================= Diff parsing helpers =========================

# We don't reuse the full-blown regex from scan_features.py here.
# For history, we use a line-oriented / small state machine that
# tolerates multi-line attributes and strange formatting, but we
# deliberately only look at crate-level inner attributes (#![...]).

RE_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_feature_inner(inner: str) -> List[str]:
    """
    Given the inside of feature(...), parse potential feature names.

    This is intentionally tolerant:
      - strips comments after //
      - splits loosely and then applies IDENT regex.
    """
    if "//" in inner:
        inner = inner.split("//", 1)[0]
    names = set()
    for ident in RE_IDENT.findall(inner):
        # In practice, feature gate names are plain identifiers
        names.add(ident)
    return sorted(names)


def collect_features_from_diff_lines(lines: List[str], sign: str) -> List[str]:
    """
    Given a list of diff lines (all with the same sign '+' or '-'),
    collect feature names from any #![feature(...)] or
    #![cfg_attr(..., feature(...))] attributes across these lines.

    We implement a small state machine to handle multi-line attributes.
    """
    assert sign in ("+", "-")
    features: List[str] = []

    pending = False
    pending_inner = ""

    for raw_line in lines:
        if not raw_line.startswith(sign):
            # This should not happen if caller groups by sign, but be tolerant.
            continue

        line = raw_line[1:].rstrip("\n")

        # If we're in the middle of a multi-line feature(...), keep appending.
        if pending:
            pending_inner += "\n" + line
            if ")" in line:
                # Close the current feature(...)
                before_close, _ = pending_inner.split(")", 1)
                feats = parse_feature_inner(before_close)
                features.extend(feats)
                pending = False
                pending_inner = ""
            continue

        # Not in pending state: look for start of feature(
        if "feature(" not in line:
            continue
        # For safety, only consider crate-level attributes.
        if "#![" not in line:
            continue

        idx = line.index("feature(")
        after = line[idx + len("feature("):]

        if ")" in after:
            before_close, _ = after.split(")", 1)
            feats = parse_feature_inner(before_close)
            features.extend(feats)
        else:
            # Attribute continues on subsequent lines
            pending = True
            pending_inner = after

    # If we end with pending=True and never saw a ')', we just drop it.
    # This is unlikely in real-world code; in worst case we miss those features
    # historically but HEAD scanner will still see them.
    return features


# ========================= Git history scanning =========================

def run_git(
    repo_path: str,
    args: List[str],
    timeout: int,
) -> Tuple[int, str, str]:
    """
    Run a git command in given repo, return (returncode, stdout, stderr).
    """
    cmd = ["git", "-C", repo_path] + args
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, "", f"git command timed out: {' '.join(cmd)}"
    except FileNotFoundError:
        return 127, "", "git not found on PATH"


def scan_repo_history(
    job: RepoHistoryJob,
    head_feature_names: List[str],
    git_timeout: int,
) -> HistoryResult:
    """
    Scan a single repo's history for feature gates.

    Returns HistoryResult with one row per (repo, feature) that ever
    appeared in the history (as per git -S 'feature(' on *.rs files).
    """
    key = job.key
    repo_path = job.path

    if not os.path.isdir(repo_path):
        return HistoryResult(key=key, success=False,
                             error=f"repo path not found: {repo_path}",
                             rows=[])

    if not os.path.isdir(os.path.join(repo_path, ".git")):
        return HistoryResult(key=key, success=False,
                             error=f"not a git repository: {repo_path}",
                             rows=[])

    # 1. Get HEAD commit hash & time
    rc, out, err = run_git(
        repo_path, ["log", "-1", "--format=%H %ct"], git_timeout
    )
    if rc != 0 or not out.strip():
        return HistoryResult(
            key=key,
            success=False,
            error=f"git log -1 failed: {err.strip()}",
            rows=[],
        )

    try:
        head_hash, head_ct_str = out.strip().split()
        head_ct = int(head_ct_str)
        head_dt = datetime.fromtimestamp(head_ct, tz=timezone.utc)
        head_iso = head_dt.isoformat(timespec="seconds").replace("+00:00", "Z")
    except Exception as e:
        return HistoryResult(
            key=key,
            success=False,
            error=f"failed to parse HEAD metadata: {e}",
            rows=[],
        )

    # 2. Run git log with pickaxe to find commits where 'feature(' count changed
    rc, out, err = run_git(
        repo_path,
        [
            "log",
            "--reverse",
            "-p",
            "-S", "feature(",
            "--format=__COMMIT__%H %ct",
            "--",
            "*.rs",
        ],
        git_timeout,
    )
    if rc != 0:
        # git log -S returns 0 even if there are no matches, so non-zero is a real error
        return HistoryResult(
            key=key,
            success=False,
            error=f"git log -S failed: {err.strip()}",
            rows=[],
        )

    if not out.strip():
        # No commits ever changed any 'feature(' occurrences: assume no historical usage.
        # (HEAD scanner will still capture current features if any.)
        return HistoryResult(key=key, success=True, error="", rows=[])

    # 3. Parse git log output: commit markers + patches
    # We'll maintain per-feature counts and first/last-seen commit info.
    feature_counts: Dict[str, int] = defaultdict(int)
    first_seen: Dict[str, Tuple[str, str]] = {}
    last_seen_remove: Dict[str, Tuple[str, str]] = {}

    lines = out.splitlines()

    current_commit: Optional[str] = None
    current_date_iso: Optional[str] = None
    plus_lines: List[str] = []
    minus_lines: List[str] = []

    def flush_commit():
        nonlocal plus_lines, minus_lines, current_commit, current_date_iso
        if current_commit is None or current_date_iso is None:
            plus_lines = []
            minus_lines = []
            return

        # Collect features added/removed in this commit
        added_feats = collect_features_from_diff_lines(plus_lines, "+")
        removed_feats = collect_features_from_diff_lines(minus_lines, "-")

        # Update per-feature counters & first/last seen
        for feat in added_feats:
            if feature_counts[feat] == 0 and feat not in first_seen:
                first_seen[feat] = (current_commit, current_date_iso)
            feature_counts[feat] += 1

        for feat in removed_feats:
            if feature_counts[feat] > 0:
                feature_counts[feat] -= 1
                if feature_counts[feat] == 0:
                    last_seen_remove[feat] = (current_commit, current_date_iso)

        plus_lines = []
        minus_lines = []

    # Iterate over lines and build commit-level diff slices
    for line in lines:
        if line.startswith("__COMMIT__"):
            # New commit header
            flush_commit()
            try:
                meta = line[len("__COMMIT__"):].strip()
                commit_hash, ct_str = meta.split()
                ct = int(ct_str)
                dt = datetime.fromtimestamp(ct, tz=timezone.utc)
                current_commit = commit_hash
                current_date_iso = dt.isoformat(timespec="seconds").replace("+00:00", "Z")
            except Exception:
                current_commit = None
                current_date_iso = None
            # Reset per-commit diff buffers
            plus_lines = []
            minus_lines = []
            continue

        # Within a commit patch: collect +/- lines that correspond to code
        if current_commit is None:
            continue

        if not line:
            continue

        # Skip file headers / hunk headers
        if line.startswith("diff ") or line.startswith("index ") \
           or line.startswith("@@ ") or line.startswith("+++ ") \
           or line.startswith("--- "):
            continue

        if line.startswith("+"):
            plus_lines.append(line)
        elif line.startswith("-"):
            minus_lines.append(line)
        else:
            # context line; ignore
            continue

    # Flush the last commit
    flush_commit()

    if not first_seen:
        # No parseable features, but git did have some changes with 'feature('.
        # Could be weird formatting; we treat as success with empty rows.
        return HistoryResult(key=key, success=True, error="", rows=[])

    # 4. Build rows for SQLite:
    # For each feature ever seen in this repo,
    #   - first_seen_commit/date from first_seen
    #   - last_seen_commit/date:
    #       - if repo HEAD still has this feature -> HEAD commit/date
    #       - else if we observed a removal -> last_seen_remove
    #       - else -> NULL
    head_feats_set = set(head_feature_names or [])
    rows: List[Tuple[str, str, str, str, Optional[str], Optional[str], int]] = []

    for feat, (fc, fd) in first_seen.items():
        still_present = 1 if feat in head_feats_set else 0
        if still_present:
            last_commit = head_hash
            last_date = head_iso
        else:
            rm_info = last_seen_remove.get(feat)
            if rm_info is not None:
                last_commit, last_date = rm_info
            else:
                last_commit, last_date = None, None

        rows.append(
            (key, feat, fc, fd, last_commit, last_date, still_present)
        )

    return HistoryResult(key=key, success=True, error="", rows=rows)


# ========================= Main =========================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Scan git history for Rust unstable feature usage and store in SQLite."
    )
    parser.add_argument(
        "--db",
        default="features.db",
        help="Path to SQLite database (default: features.db).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads (default: 4).",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=None,
        help="Limit number of repos to scan (for testing).",
    )
    parser.add_argument(
        "--only-head-nightly",
        action="store_true",
        help="Only scan repos that currently use at least one feature gate on HEAD (as per repo_features).",
    )
    parser.add_argument(
        "--git-timeout",
        type=int,
        default=600,
        help="Per-git-command timeout in seconds (default: 600).",
    )
    args = parser.parse_args(argv)

    log(f"Opening database: {args.db}")
    conn = init_db(args.db)

    # Load repos metadata
    repos = load_repos(conn)
    log(f"Loaded {len(repos)} repos from `repos` table.")

    # Load HEAD-level features for each repo
    head_features = load_head_features(conn)
    if head_features:
        log(f"Loaded HEAD features for {len(head_features)} repos from `repo_features`.")
    else:
        log("No HEAD feature data found (repo_features missing or empty). still_present will be 0 for all.")

    # Decide which repos to scan
    jobs: List[RepoHistoryJob] = list(repos.values())

    if args.only_head_nightly and head_features:
        keys_with_head = set(head_features.keys())
        jobs = [j for j in jobs if j.key in keys_with_head]
        log(f"Filtered to {len(jobs)} repos that currently use unstable features on HEAD.")
    elif args.only_head_nightly and not head_features:
        log("WARNING: --only-head-nightly set but repo_features table is missing/empty; no repos to scan.")
        jobs = []

    # Filter out repos whose path does not exist
    jobs = [j for j in jobs if os.path.isdir(j.path)]
    if args.max_repos is not None:
        jobs = jobs[: args.max_repos]

    if not jobs:
        log("No repositories to scan. Exit.")
        return

    log(f"Repos to scan for history: {len(jobs)}")

    total = len(jobs)
    completed = 0
    succ = 0
    fail = 0
    total_rows = 0
    lock = threading.Lock()

    t0 = time.time()

    log("Starting history scan worker pool ...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(
                scan_repo_history,
                job,
                head_features.get(job.key, []),
                args.git_timeout,
            ): job
            for job in jobs
        }

        try:
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    res: HistoryResult = future.result()
                except Exception as e:
                    res = HistoryResult(
                        key=job.key,
                        success=False,
                        error=f"worker exception: {e}",
                        rows=[],
                    )

                with lock:
                    completed += 1
                    if res.success:
                        succ += 1
                    else:
                        fail += 1
                    total_rows += len(res.rows)

                    # Write to DB for this repo
                    if res.success:
                        replace_repo_history(conn, res.key, res.rows)

                    state_tag = "OK" if res.success else "FAIL"
                    extra = ""
                    if res.success:
                        extra = f"features={len(res.rows)}"
                    if not res.success and res.error:
                        extra = f"error={res.error}"

                    log(
                        f"[{state_tag}] {res.key} | "
                        f"{completed}/{total} done (success={succ}, failed={fail})"
                        + (f" | {extra}" if extra else "")
                    )
        except KeyboardInterrupt:
            log("Interrupted by user (Ctrl+C). Waiting for running tasks to finish...")
            # Executor will wait for already-submitted tasks on exit.
            raise

    elapsed = max(time.time() - t0, 1e-6)
    log(
        f"History scan finished. success={succ}, failed={fail}, "
        f"total_rows={total_rows}, elapsed={elapsed:.1f}s"
    )
    conn.close()


if __name__ == "__main__":
    main()
