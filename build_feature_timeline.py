#!/usr/bin/env python3
"""
build_feature_timeline_deep.py

Advanced Rust feature timeline builder with GIT FORENSICS.
Reconstructs the true birth date of features by scanning full git history
when structural parsing fails.
"""

import argparse
import csv
import subprocess
import sys
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

# --- Regex for parsing feature files ---
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
FEATURE_ENTRY_RE = re.compile(
    r"\(\s*"
    r"(?P<status>unstable|incomplete|internal|accepted|removed|active)\s*,\s*"
    r"(?P<feature>[A-Za-z0-9_]+)\s*,\s*"
    r"\"(?P<ver>[^\"]+)\"",
    re.MULTILINE,
)

# --- Path Variants (Standard Locations) ---
PATH_VARIANTS = [
    {
        "kind": "compiler_modern",
        "files": [
            "compiler/rustc_feature/src/unstable.rs",
            "compiler/rustc_feature/src/accepted.rs",
            "compiler/rustc_feature/src/removed.rs",
        ]
    },
    {
        "kind": "librustc_middle",
        "files": [
            "src/librustc_feature/active.rs",
            "src/librustc_feature/accepted.rs",
            "src/librustc_feature/removed.rs",
        ]
    },
    {
        "kind": "libsyntax_legacy",
        "files": [
            "src/libsyntax/feature_gate.rs"
        ]
    },
]

@dataclass
class VersionInfo:
    version: str
    tag: str
    date: str

@dataclass
class StatusEntry:
    version: str
    date: str
    status: str

# --- Git Helpers ---

def run_git(repo: str, args: List[str]) -> str:
    """Run git command, strictly returning stdout."""
    cmd = ["git", "-C", repo] + args
    try:
        # We use utf-8 with replace to avoid crashing on binary garbage in old history
        return subprocess.check_output(cmd, text=True, errors="replace")
    except subprocess.CalledProcessError:
        return ""

def deep_scan_birth_date(repo: str, feature_name: str) -> Optional[str]:
    """
    NUCLEAR OPTION: Use 'git log -S' to find the absolute first time
    a feature string appeared in the entire history of the repo.
    This solves the "missing history" for renamed or moved features.
    """
    # -S looks for differences that contain the string (introduction/removal)
    # --reverse means show the oldest one first
    # --format=%cs gives YYYY-MM-DD
    # We restrict to 'src/' and 'compiler/' to avoid changelogs/tests noise if possible
    cmd = [
        "log", "-S", feature_name, "--reverse", "--format=%cs", "-n", "1",
        "--", "src", "compiler" 
    ]
    out = run_git(repo, cmd).strip()
    if not out:
        # Fallback: search everywhere if not found in src/compiler
        cmd_all = ["log", "-S", feature_name, "--reverse", "--format=%cs", "-n", "1"]
        out = run_git(repo, cmd_all).strip()
    
    return out if out else None

def collect_stable_tags(repo: str) -> List[str]:
    raw = run_git(repo, ["tag", "--list"])
    tags = []
    for line in raw.splitlines():
        name = line.strip()
        if SEMVER_RE.match(name):
            tags.append(name)
    tags.sort(key=lambda s: list(map(int, s.split('.'))))
    return tags

def get_tag_commit_date(repo: str, tag: str) -> str:
    return run_git(repo, ["log", "-1", "--format=%cs", tag]).strip()

def fetch_release_dates() -> Dict[str, str]:
    """Scrape official release dates."""
    url = "https://doc.rust-lang.org/stable/releases.html"
    print(f"[info] Fetching release dates from {url} ...", file=sys.stderr)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return {}

    pattern = re.compile(r"Version\s+(\d+\.\d+\.\d+)\s+\((\d{4}-\d{2}-\d{2})\)")
    dates = {}
    for m in pattern.finditer(html):
        dates[m.group(1)] = m.group(2)
    return dates

def git_show_file(repo: str, tag: str, path: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "show", f"{tag}:{path}"], 
            text=True, errors="replace", stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return None

def parse_features_for_tag(repo: str, tag: str) -> Dict[str, Tuple[str, str]]:
    """Scan known locations for feature definitions."""
    combined = {}
    for variant in PATH_VARIANTS:
        found_any_in_variant = False
        for path in variant["files"]:
            content = git_show_file(repo, tag, path)
            if content:
                found_any_in_variant = True
                for m in FEATURE_ENTRY_RE.finditer(content):
                    combined[m.group("feature")] = (m.group("status"), m.group("ver"))
        if found_any_in_variant:
            break
    return combined

def build_timelines(repo: str, tags: List[str], release_dates: Dict[str, str]) -> Dict[str, List[StatusEntry]]:
    version_index = {}
    for tag in tags:
        d = release_dates.get(tag) or get_tag_commit_date(repo, tag)
        version_index[tag] = VersionInfo(tag, tag, d)

    timelines: Dict[str, List[StatusEntry]] = {}
    
    # Structural Scan (The "Fast" Pass)
    sorted_vers = sorted(tags, key=lambda s: list(map(int, s.split('.'))))
    for ver in sorted_vers:
        info = version_index[ver]
        fmap = parse_features_for_tag(repo, ver)
        for feature, (status, ver_from_file) in fmap.items():
            # Correct version using in-file metadata if available
            rec_ver, rec_date = ver, info.date
            if status in ("accepted", "removed") and ver_from_file in version_index:
                rec_ver = ver_from_file
                rec_date = version_index[ver_from_file].date
            
            timelines.setdefault(feature, []).append(
                StatusEntry(rec_ver, rec_date, status)
            )

    # Dedup and Sort
    for f in timelines:
        # Sort by date
        timelines[f].sort(key=lambda x: x.date)
        # Simple dedup: keep first entry per status change ideally, 
        # but here we just sort to get min/max correctly later.

    return timelines

def write_csv(repo: str, timelines: Dict[str, List[StatusEntry]], out_path: str):
    fieldnames = [
        "feature",
        "first_appearance_version", "first_appearance_date", "first_appearance_status",
        "first_unstable_version", "first_unstable_date",
        "stabilized_version", "stabilized_date",
        "incubation_days_calculated" # New trusted field
    ]

    print(f"[info] Refining data and performing deep scan for missing histories...", file=sys.stderr)
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        total = len(timelines)
        for i, (feature, entries) in enumerate(sorted(timelines.items())):
            if i % 20 == 0:
                print(f"\r[info] Processing feature {i}/{total} ...", end="", file=sys.stderr)

            # Analyze known history
            first_any = entries[0]
            
            unstable_entries = [e for e in entries if e.status in ("unstable", "active", "internal", "incomplete")]
            stable_entries = [e for e in entries if e.status == "accepted"]
            
            first_unstable = unstable_entries[0] if unstable_entries else None
            stabilized = stable_entries[0] if stable_entries else None

            # --- THE "REAL DATA" LOGIC ---
            
            # 1. Determine "True Start Date"
            # If we saw it as unstable, great. If not, and it's stabilized, we have a gap.
            true_start_date = ""
            true_start_ver = ""
            
            if first_unstable:
                true_start_date = first_unstable.date
                true_start_ver = first_unstable.version
            else:
                # If we lack unstable history (e.g. macro_rules), we perform a DEEP SCAN
                # Only do this if it's a "broken" record (stabilized but no unstable start)
                if stabilized:
                    # Try to use first appearance first (fastest)
                    if first_any and first_any.date < stabilized.date:
                        true_start_date = first_any.date
                        true_start_ver = first_any.version
                    else:
                        # Fallback: Git Forensics
                        # This finds when the feature string was FIRST added to the repo
                        deep_date = deep_scan_birth_date(repo, feature)
                        if deep_date and deep_date <= stabilized.date:
                            true_start_date = deep_date
                            true_start_ver = "pre-history" # Version is hard to map from raw date
                        else:
                            # If git log finds nothing earlier, it truly was "born stable"
                            true_start_date = stabilized.date
                            true_start_ver = stabilized.version

            # 2. Calculate Incubation
            incubation = 0
            if true_start_date and stabilized:
                try:
                    from datetime import datetime
                    d1 = datetime.strptime(true_start_date, "%Y-%m-%d")
                    d2 = datetime.strptime(stabilized.date, "%Y-%m-%d")
                    diff = (d2 - d1).days
                    incubation = diff if diff >= 0 else 0
                except ValueError:
                    pass

            row = {
                "feature": feature,
                "first_appearance_version": first_any.version,
                "first_appearance_date": first_any.date,
                "first_appearance_status": first_any.status,
                "first_unstable_version": true_start_ver,
                "first_unstable_date": true_start_date, # This is now patched with real data
                "stabilized_version": stabilized.version if stabilized else "",
                "stabilized_date": stabilized.date if stabilized else "",
                "incubation_days_calculated": incubation
            }
            writer.writerow(row)
    
    print(f"\n[info] Done. CSV written to {out_path}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust-repo", required=True)
    ap.add_argument("--out", default="feature_timeline.csv")
    args = ap.parse_args()

    # 1. Setup
    tags = collect_stable_tags(args.rust_repo)
    dates = fetch_release_dates()
    
    # 2. Build Structural Timeline
    timelines = build_timelines(args.rust_repo, tags, dates)
    
    # 3. Write with Deep Scan
    write_csv(args.rust_repo, timelines, args.out)

if __name__ == "__main__":
    main()