#!/usr/bin/env python3
"""
scan_features.py (v2)

Scan sampled Rust GitHub repos for nightly language features via #![feature(...)]
in the current HEAD, and store results in a SQLite database for later analysis.

This version is tailored to the new sampled_crates_v2.csv header:

  crate_id,name,repository,downloads,revdeps,top_category,popularity_band,
  is_core,stratum,created_at,latest_version_created_at,latest_version_year

Usage examples:

  # Normal run: scan all repos with status=success in download_status.csv
  python3 scan_features.py \
    --csv sampled_crates_v2.csv \
    --status download_status.csv \
    --repos-dir repos \
    --db features_head_v2.db \
    --workers 8

  # Small test on the first 5 repos
  python3 scan_features.py \
    --csv sampled_crates_v2.csv \
    --status download_status.csv \
    --repos-dir repos \
    --db features_head_v2.db \
    --workers 4 \
    --max-repos 5

The DB schema is kept simple but rich enough for later statistics.
"""

import argparse
import csv
import os
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ========================= Logging =========================

def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


# ========================= Data structures =========================

@dataclass
class RepoJob:
    key: str                 # "owner/repo"
    owner: str
    repo: str
    local_path: str          # local Git repo path

    crate_ids: List[str]
    crate_names: List[str]

    downloads_sum: int
    revdeps_sum: int

    top_categories: List[str]
    popularity_bands: List[str]
    is_core: bool
    strata: List[str]


@dataclass
class ScanResult:
    key: str
    success: bool
    error: str
    feature_counts: Dict[str, int]
    feature_files: Dict[str, int]   # feature -> number of .rs files containing it
    example_paths: Dict[str, str]   # feature -> one sample relative file path


# ========================= GitHub URL parsing =========================

def parse_github_repo(repository: str) -> Optional[Tuple[str, str]]:
    """
    Parse GitHub URLs into (owner, repo).

    Supports:
      - https://github.com/owner/repo.git
      - https://github.com/owner/repo
      - git@github.com:owner/repo.git
      - github.com/owner/repo

    Returns None if not a GitHub repo or too weird to parse.
    """
    if not repository:
        return None
    s = repository.strip()

    # SSH form: git@github.com:owner/repo.git
    if s.startswith("git@github.com:"):
        tail = s[len("git@github.com:"):]
        tail = tail.split("#", 1)[0].split("?", 1)[0]
        if tail.endswith(".git"):
            tail = tail[:-4]
        parts = tail.strip("/").split("/")
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1]
        return owner, repo

    lower = s.lower()
    if "github.com" not in lower:
        return None

    # e.g. "github.com/owner/repo"
    if not (s.startswith("http://") or s.startswith("https://")
            or s.startswith("git://") or s.startswith("ssh://")):
        s = "https://" + s.lstrip("/")

    parsed = urlparse(s)
    host = parsed.netloc.lower()
    path = parsed.path

    if "github.com" not in host:
        # maybe we got "github.com/owner/repo" as path
        if parsed.netloc == "" and parsed.path.startswith("github.com/"):
            path = parsed.path[len("github.com") + 1:]
        else:
            return None

    path_str = path.strip("/")
    parts = path_str.split("/")
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


# ========================= Load sampled CSV & status =========================

def load_jobs_from_sample(csv_path: str, repos_dir: str) -> Dict[str, RepoJob]:
    """
    Load sampled_crates_v2.csv, group rows by GitHub repo, and build RepoJob objects.

    Expected CSV header (at least these columns):

      crate_id,name,repository,downloads,revdeps,
      top_category,popularity_band,is_core,stratum,...

    Returns: dict key -> RepoJob
    """
    jobs: Dict[str, RepoJob] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = [
            "crate_id",
            "name",
            "repository",
            "downloads",
            "revdeps",
            "top_category",
            "popularity_band",
            "is_core",
            "stratum",
        ]
        for c in required_cols:
            if c not in reader.fieldnames:
                raise RuntimeError(f"sample CSV missing column: {c}")

        for row in reader:
            repo_url = (row.get("repository") or "").strip()
            parsed = parse_github_repo(repo_url)
            if parsed is None:
                # non-GitHub or weird URL -> skip
                continue
            owner, repo = parsed
            key = f"{owner}/{repo}"
            local_path = os.path.join(repos_dir, owner, repo)

            crate_id = (row.get("crate_id") or "").strip()
            crate_name = (row.get("name") or "").strip()

            def _as_int(s: str) -> int:
                s = (s or "").strip()
                if not s:
                    return 0
                try:
                    return int(s)
                except ValueError:
                    # in case of weird stuff, fall back to 0
                    return 0

            downloads = _as_int(row.get("downloads") or "0")
            revdeps = _as_int(row.get("revdeps") or "0")

            top_category = (row.get("top_category") or "uncategorized").strip() or "uncategorized"
            pop_band = (row.get("popularity_band") or "unknown").strip() or "unknown"
            is_core_flag = (row.get("is_core") or "0").strip().lower()
            is_core = is_core_flag in ("1", "true", "yes", "y")
            stratum = (row.get("stratum") or "non-core").strip() or "non-core"

            if key not in jobs:
                jobs[key] = RepoJob(
                    key=key,
                    owner=owner,
                    repo=repo,
                    local_path=local_path,
                    crate_ids=[crate_id] if crate_id else [],
                    crate_names=[crate_name] if crate_name else [],
                    downloads_sum=downloads,
                    revdeps_sum=revdeps,
                    top_categories=[top_category],
                    popularity_bands=[pop_band],
                    is_core=is_core,
                    strata=[stratum],
                )
            else:
                job = jobs[key]
                if crate_id and crate_id not in job.crate_ids:
                    job.crate_ids.append(crate_id)
                if crate_name and crate_name not in job.crate_names:
                    job.crate_names.append(crate_name)

                job.downloads_sum += downloads
                job.revdeps_sum += revdeps

                if top_category not in job.top_categories:
                    job.top_categories.append(top_category)
                if pop_band not in job.popularity_bands:
                    job.popularity_bands.append(pop_band)
                if stratum not in job.strata:
                    job.strata.append(stratum)

                job.is_core = job.is_core or is_core

    return jobs


def load_status(status_path: str) -> Dict[str, str]:
    """
    Load key -> status mapping from download_status.csv.

    If the file does not exist, return an empty dict (meaning: no status filter).
    """
    status: Dict[str, str] = {}

    if not os.path.exists(status_path):
        return status

    with open(status_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("key") or "").strip()
            st = (row.get("status") or "").strip()
            if key:
                status[key] = st

    return status


# ========================= SQLite schema & helpers =========================

def init_db(db_path: str) -> sqlite3.Connection:
    """
    Initialize SQLite connection and schema.

    Tables:

      repos(
        key TEXT PRIMARY KEY,
        owner TEXT,
        name TEXT,
        path TEXT,
        crate_ids TEXT,
        crate_names TEXT,
        downloads_sum INTEGER,
        revdeps_sum INTEGER,
        top_categories TEXT,
        popularity_bands TEXT,
        is_core INTEGER,
        strata TEXT
      )

      repo_features(
        key TEXT,
        feature_name TEXT,
        attr_count INTEGER,
        file_count INTEGER,
        example_paths TEXT,
        PRIMARY KEY (key, feature_name),
        FOREIGN KEY (key) REFERENCES repos(key)
      )
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS repos (
          key               TEXT PRIMARY KEY,
          owner             TEXT,
          name              TEXT,
          path              TEXT,
          crate_ids         TEXT,
          crate_names       TEXT,
          downloads_sum     INTEGER,
          revdeps_sum       INTEGER,
          top_categories    TEXT,
          popularity_bands  TEXT,
          is_core           INTEGER,
          strata            TEXT
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS repo_features (
          key           TEXT,
          feature_name  TEXT,
          attr_count    INTEGER,
          file_count    INTEGER,
          example_paths TEXT,
          PRIMARY KEY (key, feature_name),
          FOREIGN KEY (key) REFERENCES repos(key)
        );
        """
    )

    conn.commit()
    return conn


def upsert_repo(conn: sqlite3.Connection, job: RepoJob) -> None:
    """
    Insert or replace a row in repos for the given RepoJob.
    """
    crate_ids = ",".join(sorted({c for c in job.crate_ids if c}))
    crate_names = ",".join(sorted({c for c in job.crate_names if c}))
    top_categories = ",".join(sorted({c for c in job.top_categories if c}))
    pop_bands = ",".join(sorted({c for c in job.popularity_bands if c}))
    strata = ",".join(sorted({s for s in job.strata if s}))
    is_core_int = 1 if job.is_core else 0

    conn.execute(
        """
        INSERT OR REPLACE INTO repos
          (key, owner, name, path, crate_ids, crate_names,
           downloads_sum, revdeps_sum,
           top_categories, popularity_bands, is_core, strata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            job.key,
            job.owner,
            job.repo,
            job.local_path,
            crate_ids,
            crate_names,
            job.downloads_sum,
            job.revdeps_sum,
            top_categories,
            pop_bands,
            is_core_int,
            strata,
        ),
    )


def update_repo_features(
    conn: sqlite3.Connection,
    key: str,
    feature_counts: Dict[str, int],
    feature_files: Dict[str, int],
    example_paths: Dict[str, str],
) -> None:
    """
    Update repo_features for a repo:
      - delete old rows
      - insert new rows from current scan
    """
    conn.execute("DELETE FROM repo_features WHERE key = ?;", (key,))

    rows = []
    for feat, cnt in feature_counts.items():
        file_cnt = feature_files.get(feat, 0)
        ex_path = example_paths.get(feat, "")
        rows.append((key, feat, cnt, file_cnt, ex_path))

    if rows:
        conn.executemany(
            """
            INSERT INTO repo_features (key, feature_name, attr_count, file_count, example_paths)
            VALUES (?, ?, ?, ?, ?);
            """,
            rows,
        )


# ========================= Feature scanning logic =========================

# direct feature gate: #![feature(allocator_api, generic_const_exprs)]
RE_FEATURE_DIRECT = re.compile(
    r"#!\s*\[\s*feature\s*\((?P<inner>.*?)\)\s*\]",
    re.DOTALL,
)

# cfg_attr(..., feature(...)) form:
#   #![cfg_attr(feature = "nightly", feature(specialization))]
RE_FEATURE_CFG_ATTR = re.compile(
    r"#!\s*\[\s*cfg_attr\([^,\)]*,\s*feature\s*\((?P<inner>.*?)\)\s*\)\s*\]",
    re.DOTALL,
)

RE_FEATURE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def extract_features_from_text(text: str) -> Dict[str, int]:
    """
    Extract feature names and their counts from one Rust source file.
    Only looks at attributes of the form #![feature(...)] and
    #![cfg_attr(..., feature(...))].
    """
    counts: Dict[str, int] = defaultdict(int)

    def _process_inner(inner: str):
        # inner ~ "allocator_api, generic_const_exprs"
        for part in inner.split(","):
            token = part.strip()
            if not token:
                continue
            if "//" in token:
                token = token.split("//", 1)[0].strip()
            if not token:
                continue
            if not RE_FEATURE_NAME.match(token):
                continue
            counts[token] += 1

    for m in RE_FEATURE_DIRECT.finditer(text):
        _process_inner(m.group("inner"))

    for m in RE_FEATURE_CFG_ATTR.finditer(text):
        _process_inner(m.group("inner"))

    return counts


def scan_single_repo(job: RepoJob) -> ScanResult:
    """
    Worker entry: scan a single repo and return ScanResult.
    """
    feature_counts: Dict[str, int] = defaultdict(int)
    feature_files: Dict[str, int] = defaultdict(int)
    feature_example_path: Dict[str, str] = {}

    if not os.path.isdir(job.local_path):
        return ScanResult(
            key=job.key,
            success=False,
            error=f"local path not found: {job.local_path}",
            feature_counts={},
            feature_files={},
            example_paths={},
        )

    try:
        for root, dirs, files in os.walk(job.local_path):
            # prune big/irrelevant dirs to speed up scan
            dirs[:] = [
                d for d in dirs
                if d not in (".git", "target", "node_modules", "dist", "build")
            ]

            for name in files:
                if not name.endswith(".rs"):
                    continue

                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, job.local_path)

                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    # ignore single-file errors
                    continue

                file_counts = extract_features_from_text(text)
                if not file_counts:
                    continue

                for feat, cnt in file_counts.items():
                    feature_counts[feat] += cnt
                    feature_files[feat] += 1
                    feature_example_path.setdefault(feat, rel_path)

        if not feature_counts:
            # no feature gates found; treat as success with empty result
            return ScanResult(
                key=job.key,
                success=True,
                error="",
                feature_counts={},
                feature_files={},
                example_paths={},
            )

        return ScanResult(
            key=job.key,
            success=True,
            error="",
            feature_counts=dict(feature_counts),
            feature_files=dict(feature_files),
            example_paths=feature_example_path,
        )
    except Exception as e:
        return ScanResult(
            key=job.key,
            success=False,
            error=str(e),
            feature_counts={},
            feature_files={},
            example_paths={},
        )


# ========================= Main =========================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Scan Rust repos for #![feature(...)] usage (HEAD) and store results in SQLite."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to sampled_crates_v2.csv generated by the sampling script.",
    )
    parser.add_argument(
        "--status",
        default="download_status.csv",
        help="Path to download_status.csv (default: download_status.csv).",
    )
    parser.add_argument(
        "--repos-dir",
        default="repos",
        help="Directory where Git repos are stored (default: ./repos).",
    )
    parser.add_argument(
        "--db",
        default="features_head_v2.db",
        help="SQLite database path (default: features_head_v2.db).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers (default: 8).",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=None,
        help="Max number of repos to scan (for quick tests).",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also scan repos whose download status is 'failed' if local path exists.",
    )
    args = parser.parse_args(argv)

    # 1. Load jobs & status
    log(f"Loading jobs from {args.csv} ...")
    jobs = load_jobs_from_sample(args.csv, args.repos_dir)
    log(f"Total unique GitHub repos in sample: {len(jobs)}")

    status_map = load_status(args.status)
    if status_map:
        log(f"Loaded download status for {len(status_map)} repos from {args.status}")
    else:
        log("Status file not found or empty. Will scan based only on local paths.")

    # 2. Decide which repos to scan
    to_scan: List[RepoJob] = []
    for key, job in jobs.items():
        st = status_map.get(key, "unknown")
        if status_map:
            if st == "success":
                to_scan.append(job)
            elif args.include_failed and os.path.isdir(job.local_path):
                to_scan.append(job)
        else:
            # no status file: scan whatever exists locally
            if os.path.isdir(job.local_path):
                to_scan.append(job)

    if args.max_repos is not None:
        to_scan = to_scan[: args.max_repos]

    log(f"Repos to scan in this run: {len(to_scan)}")
    if not to_scan:
        log("Nothing to scan. Exit.")
        return

    # 3. Init DB & upsert repos metadata
    conn = init_db(args.db)
    for job in to_scan:
        upsert_repo(conn, job)
    conn.commit()
    log(f"Repo metadata upserted into {args.db}")

    # 4. Run multi-threaded scan
    total = len(to_scan)
    completed = 0
    succ = 0
    fail = 0
    total_features_entries = 0

    lock = threading.Lock()
    log("Starting feature scan worker pool ...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {executor.submit(scan_single_repo, job): job for job in to_scan}

        try:
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    res: ScanResult = future.result()
                except Exception as e:
                    res = ScanResult(
                        key=job.key,
                        success=False,
                        error=f"worker exception: {e}",
                        feature_counts={},
                        feature_files={},
                        example_paths={},
                    )

                with lock:
                    completed += 1
                    if res.success:
                        succ += 1
                    else:
                        fail += 1
                    total_features_entries += len(res.feature_counts)

                    if res.success and res.feature_counts:
                        update_repo_features(
                            conn,
                            res.key,
                            res.feature_counts,
                            res.feature_files,
                            res.example_paths,
                        )
                        conn.commit()

                    feat_info = f"{len(res.feature_counts)} features" if res.success else "0 features"
                    state_tag = "OK" if res.success else "FAIL"
                    log(
                        f"[{state_tag}] {res.key} | "
                        f"{completed}/{total} done (success={succ}, failed={fail}) | "
                        f"{feat_info}"
                        + (f" | error={res.error}" if (not res.success and res.error) else "")
                    )
        except KeyboardInterrupt:
            log("Interrupted by user (Ctrl+C). Waiting for running tasks to finish...")
            raise

    elapsed = max(time.time() - t0, 1e-6)
    log(
        f"Scan finished. success={succ}, failed={fail}, "
        f"total_features_entries={total_features_entries}, elapsed={elapsed:.1f}s"
    )
    conn.close()


if __name__ == "__main__":
    main()
