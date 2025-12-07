import argparse
import concurrent.futures
import csv
import os
import pathlib
import subprocess
from typing import List, Tuple

ROOT = pathlib.Path("repos")
OUTPUT_CSV = pathlib.Path("repo_commit_counts.csv")


def find_git_repos(root: pathlib.Path) -> List[pathlib.Path]:
    """
    Repos are stored as repos/<owner>/<repo>. Walk exactly two levels deep
    and return paths that contain a .git directory.
    """
    repos: List[pathlib.Path] = []
    for owner_dir in root.iterdir():
        if not owner_dir.is_dir():
            continue
        for repo_dir in owner_dir.iterdir():
            git_dir = repo_dir / ".git"
            if git_dir.is_dir():
                repos.append(repo_dir)
    return repos


def count_commits(repo_dir: pathlib.Path) -> int:
    """
    Count all commits in the repo (all refs) to maximize the number.
    """
    out = subprocess.check_output(
        ["git", "rev-list", "--count", "--all"],
        cwd=repo_dir,
        text=True,
    )
    return int(out.strip() or "0")


def count_rust_loc(repo_dir: pathlib.Path) -> int:
    """
    Rough Rust LOC: sum lines over tracked *.rs files.
    Uses git ls-files so we only count tracked Rust sources.
    """
    try:
        listed = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.rs"],
            cwd=repo_dir,
            text=True,
        )
    except subprocess.CalledProcessError:
        return 0

    total = 0
    for rel in listed.split("\0"):
        if not rel:
            continue
        path = repo_dir / rel
        try:
            with path.open("rb") as fh:
                for _ in fh:
                    total += 1
        except OSError:
            continue
    return total


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-loc", action="store_true", help="Also compute Rust LOC per repo")
    default_workers = min(8, os.cpu_count() or 4)
    ap.add_argument("--workers", type=int, default=default_workers, help="Thread workers for parallel counting")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    rows: List[Tuple[str, int, int]] = []
    total_commits = 0
    total_loc = 0

    repo_dirs = find_git_repos(ROOT)
    print(f"Using {args.workers} workers over {len(repo_dirs)} repos...")

    def process(repo_dir: pathlib.Path) -> Tuple[str, int, int]:
        commit_count = count_commits(repo_dir)
        loc = count_rust_loc(repo_dir) if args.with_loc else 0
        repo_key = f"{repo_dir.parent.name}/{repo_dir.name}"
        return repo_key, commit_count, loc

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process, rd): rd for rd in repo_dirs}
        for fut in concurrent.futures.as_completed(futures):
            repo_dir = futures[fut]
            try:
                repo_key, commit_count, loc = fut.result()
            except subprocess.CalledProcessError as e:
                print(f"[warn] Failed to count {repo_dir}: {e}")
                continue
            rows.append((repo_key, commit_count, loc))
            total_commits += commit_count
            total_loc += loc

    print(f"Repos counted: {len(rows)}")
    print(f"Total commits (sum of all repos): {total_commits}")
    if args.with_loc:
        print(f"Total Rust LOC (sum of tracked *.rs files): {total_loc}")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["repo", "commit_count"]
        if args.with_loc:
            header.append("rust_loc")
        writer.writerow(header)
        for repo_key, commit_count, loc in rows:
            row = [repo_key, commit_count]
            if args.with_loc:
                row.append(loc)
            writer.writerow(row)
    print(f"Wrote per-repo counts to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
