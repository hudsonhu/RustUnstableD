#!/usr/bin/env python3
"""
build_feature_timeline.py

Build a per-feature lifecycle timeline for Rust language feature gates
by scanning the rust-lang/rust git repository.

For each feature mentioned in rustc_feature's {active/unstable, accepted, removed}
lists across all stable releases, this script computes:

- first_appearance_* : first release in which the feature appears in any of
  {unstable, incomplete, internal, accepted, removed}.
- first_unstable_*   : first release where the feature is in an "unstable-like"
  status (unstable / incomplete / internal).
- last_unstable_*    : last release where the feature is still unstable-like.
- stabilized_*       : first release where the feature is marked accepted.
- removed_*          : first release where the feature is marked removed.
- latest_*           : most recent status we see in the scanned releases.

Release dates are taken from:
    https://doc.rust-lang.org/stable/releases.html
when available, and otherwise fall back to the git tag commit date.

This is meant for offline analysis in your data challenge project.
"""

import argparse
import csv
import subprocess
import sys
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# Matches a feature line inside {unstable,active,accepted,removed}.rs.
# Example line shapes:
#   (unstable, box_patterns, "1.0.0", Some(29641)),
#   (internal, lang_items, "1.0.0", None),
#   (removed, generic_associated_types_extended, "1.85.0", Some(95451), Some("...")),
#
# We *only* care about status, feature name, and the version string literal.
FEATURE_ENTRY_RE = re.compile(
    r"\(\s*"
    r"(?P<status>unstable|incomplete|internal|accepted|removed)\s*,\s*"
    r"(?P<feature>[A-Za-z0-9_]+)\s*,\s*"
    r"\"(?P<ver>[^\"]+)\"",
    re.MULTILINE,
)


@dataclass
class VersionInfo:
    version: str          # "1.75.0"
    tag: str              # tag name, usually same as version
    date: str             # "YYYY-MM-DD" (official or approximated)


@dataclass
class StatusEntry:
    version: str          # "1.75.0"
    date: str             # "YYYY-MM-DD"
    status: str           # "unstable" / "incomplete" / "internal" / "accepted" / "removed"

    def is_unstable_like(self) -> bool:
        return self.status in ("unstable", "incomplete", "internal")


def run_git(repo: str, args: List[str]) -> str:
    """Run a git command in the given repo and return stdout."""
    cmd = ["git", "-C", repo] + args
    try:
        out = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[git error] {' '.join(cmd)}", file=sys.stderr)
        raise
    return out


def parse_semver(v: str) -> Tuple[int, int, int]:
    m = SEMVER_RE.match(v)
    if not m:
        raise ValueError(f"Not a semver tag: {v}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def collect_stable_tags(repo: str) -> List[str]:
    """Return a sorted list of stable tags like '1.75.0'."""
    raw = run_git(repo, ["tag", "--list"])
    tags = []
    for line in raw.splitlines():
        name = line.strip()
        if SEMVER_RE.match(name):
            tags.append(name)
    tags.sort(key=parse_semver)
    return tags


def get_tag_commit_date(repo: str, tag: str) -> str:
    """
    Return the commit date (YYYY-MM-DD) of the given tag.

    This is only used as a fallback when we don't have an official release date
    from the docs.
    """
    out = run_git(repo, ["log", "-1", "--format=%cs", tag])
    return out.strip()


def fetch_release_dates() -> Dict[str, str]:
    """
    Fetch official Rust release dates from stable documentation.

    We parse headings of the form:
        "Version 1.75.0 (2023-12-28)"
    from https://doc.rust-lang.org/stable/releases.html

    Returns:
        { "1.75.0": "2023-12-28", ... }
    """
    url = "https://doc.rust-lang.org/stable/releases.html"
    print(f"[info] Fetching release dates from {url} ...", file=sys.stderr)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"[warn] Failed to fetch release dates: {e}", file=sys.stderr)
        print("[warn] Will fall back to git tag dates.", file=sys.stderr)
        return {}

    # This is deliberately simple and robust: just look for
    #   Version X.Y.Z (YYYY-MM-DD)
    pattern = re.compile(
        r"Version\s+(\d+\.\d+\.\d+)\s+\((\d{4}-\d{2}-\d{2})\)"
    )
    dates: Dict[str, str] = {}
    for m in pattern.finditer(html):
        ver, date = m.group(1), m.group(2)
        dates[ver] = date

    if not dates:
        print("[warn] Could not find any 'Version X.Y.Z (YYYY-MM-DD)' headings in releases.html.", file=sys.stderr)
    else:
        print(f"[info] Parsed {len(dates)} release dates from stable docs.", file=sys.stderr)

    return dates


def git_show_file(repo: str, tag: str, path: str) -> Optional[str]:
    """Return the file contents at tag:path, or None if path does not exist at that tag."""
    spec = f"{tag}:{path}"
    try:
        out = run_git(repo, ["show", spec])
        return out
    except subprocess.CalledProcessError:
        # Path doesn't exist at that tag
        return None


def parse_feature_file(text: str) -> Dict[str, Tuple[str, str]]:
    """
    Parse a rustc_feature {active/unstable,accepted,removed}.rs file.

    Returns:
        { feature_name: (status, ver_string_from_file) }
    """
    result: Dict[str, Tuple[str, str]] = {}
    for m in FEATURE_ENTRY_RE.finditer(text):
        status = m.group("status")
        feature = m.group("feature")
        ver = m.group("ver")
        # Entries may appear multiple times in weird historical states; last wins.
        result[feature] = (status, ver)
    return result


# Two historical layouts:
#   - Old: src/librustc_feature/{active,accepted,removed}.rs
#   - New: compiler/rustc_feature/src/{unstable,accepted,removed}.rs
PATH_VARIANTS = [
    {
        "kind": "compiler",
        "unstable": "compiler/rustc_feature/src/unstable.rs",
        "accepted": "compiler/rustc_feature/src/accepted.rs",
        "removed": "compiler/rustc_feature/src/removed.rs",
    },
    {
        "kind": "librustc",
        "unstable": "src/librustc_feature/active.rs",
        "accepted": "src/librustc_feature/accepted.rs",
        "removed": "src/librustc_feature/removed.rs",
    },
    {
        "kind": "libsyntax_legacy",
        "unstable": "src/libsyntax/feature_gate.rs", 
        "accepted": "src/libsyntax/feature_gate.rs", # 早期往往都在一个文件里
        "removed": "src/libsyntax/feature_gate.rs",
    },
]


def parse_features_for_tag(repo: str, tag: str) -> Dict[str, Tuple[str, str]]:
    """
    For a given tag, read the appropriate rustc_feature files and return:

        { feature_name: (status, ver_string_from_file) }

    Status is one of: unstable, incomplete, internal, accepted, removed.
    """
    # Try the modern layout first, then the older librustc_feature layout.
    any_found = False
    combined: Dict[str, Tuple[str, str]] = {}

    for variant in PATH_VARIANTS:
        unstable_path = variant["unstable"]
        accepted_path = variant["accepted"]
        removed_path = variant["removed"]

        unstable_text = git_show_file(repo, tag, unstable_path)
        accepted_text = git_show_file(repo, tag, accepted_path)
        removed_text = git_show_file(repo, tag, removed_path)

        if not any([unstable_text, accepted_text, removed_text]):
            # This layout isn't present at this tag; try next variant.
            continue

        any_found = True

        if unstable_text:
            d = parse_feature_file(unstable_text)
            # statuses here are unstable/incomplete/internal
            combined.update(d)

        if accepted_text:
            d = parse_feature_file(accepted_text)
            # statuses here are accepted
            combined.update(d)

        if removed_text:
            d = parse_feature_file(removed_text)
            # statuses here are removed
            combined.update(d)

        # If this variant worked, we don't need to try the other one.
        break

    if not any_found:
        # Very old Rust versions may not have rustc_feature at all; we skip them.
        return {}

    return combined


def build_version_index(
    repo: str,
    tags: List[str],
    release_dates: Dict[str, str],
) -> Dict[str, VersionInfo]:
    """
    Build mapping version -> VersionInfo, using official release date when available.
    """
    idx: Dict[str, VersionInfo] = {}
    for tag in tags:
        ver = tag  # tags are already like "1.75.0"
        date = release_dates.get(ver)
        if date is None:
            # Fallback to commit date
            date = get_tag_commit_date(repo, tag)
        idx[ver] = VersionInfo(version=ver, tag=tag, date=date)
    return idx


def build_feature_timelines(
    repo: str,
    version_index: Dict[str, VersionInfo],
) -> Dict[str, List[StatusEntry]]:
    """
    For each feature, build a chronologically ordered list of StatusEntry.
    FIXED: Uses the version string defined INSIDE the file for accepted/removed events.
    """
    timelines: Dict[str, List[StatusEntry]] = {}

    # Process tags in ascending version order
    sorted_versions = sorted(version_index.keys(), key=parse_semver)

    for ver in sorted_versions:
        info = version_index[ver]
        tag = info.tag
        # print(f"[info] Scanning features for {ver} ...", file=sys.stderr)
        feature_map = parse_features_for_tag(repo, tag)
        if not feature_map:
            continue

        for feature, (status, ver_from_file) in feature_map.items():
            # --- 修复开始 ---
            # 默认使用当前扫描的 tag 版本和日期
            record_ver = ver
            record_date = info.date
            
            # 如果文件里明确写了版本号（通常 accepted/removed 会写），
            # 且那个版本号在我们的索引里存在，则使用历史真实版本。
            # 这能解决 "2020-01-30" 堆积问题。
            if status in ("accepted", "removed") and ver_from_file in version_index:
                historical_info = version_index[ver_from_file]
                record_ver = historical_info.version
                record_date = historical_info.date
            
            # 注意：如果 ver_from_file 是 "1.0.0" 但当前扫描的是 1.41.0，
            # 我们可能会多次（在每个版本里）添加同一个 accepted 记录。
            # 为了避免重复和混乱，可以在后续步骤去重，或者这里简单处理。
            # 但既然我们最后只取 find_first_with_status，多次添加应该不影响最终 CSV 的 first/last 判定。
            # 实际上，保留每个版本的“快照”也是一种策略。
            
            entry = StatusEntry(version=record_ver, date=record_date, status=status)
            timelines.setdefault(feature, []).append(entry)
            # --- 修复结束 ---

    # 简单的后处理：对每个 feature 的 timeline 按日期/版本排序
    # 因为引入了历史版本，原本的 append 顺序可能不再是严格的时间序
    for feature in timelines:
        timelines[feature].sort(key=lambda x: parse_semver(x.version))

    return timelines


def choose_first(entries: List[StatusEntry]) -> Optional[StatusEntry]:
    return entries[0] if entries else None


def choose_last(entries: List[StatusEntry]) -> Optional[StatusEntry]:
    return entries[-1] if entries else None


def find_first_with_status(
    entries: List[StatusEntry],
    wanted_statuses: Tuple[str, ...],
) -> Optional[StatusEntry]:
    for e in entries:
        if e.status in wanted_statuses:
            return e
    return None


def find_last_with_status(
    entries: List[StatusEntry],
    wanted_statuses: Tuple[str, ...],
) -> Optional[StatusEntry]:
    last: Optional[StatusEntry] = None
    for e in entries:
        if e.status in wanted_statuses:
            last = e
    return last


def write_csv(
    timelines: Dict[str, List[StatusEntry]],
    out_path: str,
) -> None:
    """
    Write the feature lifecycle summary to CSV.

    Columns:

    - feature
    - first_appearance_version, first_appearance_date, first_appearance_status
    - first_unstable_version, first_unstable_date
    - last_unstable_version, last_unstable_date
    - stabilized_version, stabilized_date
    - removed_version, removed_date
    - latest_status, latest_version, latest_version_date
    """
    fieldnames = [
        "feature",
        "first_appearance_version",
        "first_appearance_date",
        "first_appearance_status",
        "first_unstable_version",
        "first_unstable_date",
        "last_unstable_version",
        "last_unstable_date",
        "stabilized_version",
        "stabilized_date",
        "removed_version",
        "removed_date",
        "latest_status",
        "latest_version",
        "latest_version_date",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for feature in sorted(timelines.keys()):
            entries = timelines[feature]
            # entries are already in ascending version order
            first_any = choose_first(entries)
            latest = choose_last(entries)

            unstable_like = ("unstable", "incomplete", "internal")

            first_unstable = find_first_with_status(entries, unstable_like)
            last_unstable = find_last_with_status(entries, unstable_like)
            stabilized = find_first_with_status(entries, ("accepted",))
            removed = find_first_with_status(entries, ("removed",))

            row = {
                "feature": feature,
                # First appearance in *any* status
                "first_appearance_version": first_any.version if first_any else "",
                "first_appearance_date": first_any.date if first_any else "",
                "first_appearance_status": first_any.status if first_any else "",
                # First / last unstable-like
                "first_unstable_version": first_unstable.version if first_unstable else "",
                "first_unstable_date": first_unstable.date if first_unstable else "",
                "last_unstable_version": last_unstable.version if last_unstable else "",
                "last_unstable_date": last_unstable.date if last_unstable else "",
                # Stabilization and removal
                "stabilized_version": stabilized.version if stabilized else "",
                "stabilized_date": stabilized.date if stabilized else "",
                "removed_version": removed.version if removed else "",
                "removed_date": removed.date if removed else "",
                # Latest known status
                "latest_status": latest.status if latest else "",
                "latest_version": latest.version if latest else "",
                "latest_version_date": latest.date if latest else "",
            }

            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a Rust feature-gate lifecycle CSV from rust-lang/rust.",
    )
    ap.add_argument(
        "--rust-repo",
        required=True,
        help="Path to a local clone of https://github.com/rust-lang/rust",
    )
    ap.add_argument(
        "--out",
        default="feature_timeline.csv",
        help="Output CSV path (default: feature_timeline.csv)",
    )
    args = ap.parse_args()

    repo = args.rust_repo
    out_path = args.out

    # 1) Collect stable tags
    print("[info] Collecting stable tags from git ...", file=sys.stderr)
    tags = collect_stable_tags(repo)
    if not tags:
        print("[error] No semver-looking tags found in the repo.", file=sys.stderr)
        sys.exit(1)
    print(f"[info] Found {len(tags)} stable-looking tags.", file=sys.stderr)

    # 2) Fetch official release dates (best-effort)
    release_dates = fetch_release_dates()

    # 3) Build version index with dates
    version_index = build_version_index(repo, tags, release_dates)
    print(f"[info] Built version index for {len(version_index)} versions.", file=sys.stderr)

    # 4) Build per-feature timelines
    timelines = build_feature_timelines(repo, version_index)
    print(f"[info] Collected timelines for {len(timelines)} features.", file=sys.stderr)

    # 5) Write CSV
    write_csv(timelines, out_path)
    print(f"[info] Wrote feature timeline CSV to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
