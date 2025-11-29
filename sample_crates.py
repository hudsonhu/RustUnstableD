#!/usr/bin/env python3
"""
sample_crates.py

Two-layer sampling for Rust crates:

1) Core stratum (infrastructure crates, selected with certainty):
   - Top N crates by reverse dependencies (default: 300)
   - Plus optional extra names from a core list CSV (--core-list)

2) Non-core stratum (all remaining crates):
   - Filter by minimum downloads and recency
   - Stratified random sampling by (top_category, popularity_band)
   - Popularity bands are based on non-core downloads quantiles (p50, p90)

Output: a CSV of sampled crates, including core + non-core, with labels
and a `repository` column so that download_repos.py can clone them.

Example:

    python3 sample_crates.py \
      --dump ./db-dump.tar.gz \
      --out ./sampled_crates_v2.csv \
      --target-size 1500 \
      --core-top-revdeps 300 \
      --min-downloads-noncore 100 \
      --min-latest-year-noncore 2015 \
      --seed 42

"""

import argparse
import csv
import math
import random
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Avoid "field larger than field limit" on crates.csv descriptions
csv.field_size_limit(2**31 - 1)


@dataclass
class CrateMeta:
    crate_id: int
    name: str
    repository: str
    created_at: str
    latest_version_created_at: str
    latest_version_year: Optional[int]
    downloads: int
    revdeps: int
    top_category: str


def log(msg: str):
    print(msg, flush=True)


def find_member(tf: tarfile.TarFile, basename: str) -> tarfile.TarInfo:
    """
    Find a CSV file in the tarball by basename.
    The dump may put files in the root or in a subdir (e.g. 'data/crates.csv').
    """
    for m in tf.getmembers():
        if m.name.endswith("/" + basename) or m.name == basename:
            return m
    raise FileNotFoundError(f"Could not find {basename} in tarball")


def load_crates(tf: tarfile.TarFile) -> Dict[int, Dict]:
    """
    Load crates.csv: {id: {"name": str, "created_at": str, "repository": str}}
    """
    log("[sample_crates] Reading crates.csv ...")
    crates_member = find_member(tf, "crates.csv")
    crates: Dict[int, Dict] = {}
    with tf.extractfile(crates_member) as f:
        reader = csv.DictReader((line.decode("utf-8") for line in f))
        for row in reader:
            try:
                cid = int(row["id"])
            except (KeyError, ValueError):
                continue
            crates[cid] = {
                "name": row.get("name", ""),
                "created_at": row.get("created_at", ""),
                "repository": row.get("repository", "") or "",
            }
    return crates


def load_downloads(tf: tarfile.TarFile) -> Dict[int, int]:
    """
    Load crate_downloads.csv: sum downloads per crate_id.
    """
    log("[sample_crates] Reading crate_downloads.csv ...")
    member = find_member(tf, "crate_downloads.csv")
    downloads: Dict[int, int] = defaultdict(int)
    with tf.extractfile(member) as f:
        reader = csv.DictReader((line.decode("utf-8") for line in f))
        for row in reader:
            try:
                cid = int(row["crate_id"])
                d = int(row["downloads"])
            except (KeyError, ValueError):
                continue
            downloads[cid] += d
    return downloads


def load_reverse_deps(tf: tarfile.TarFile) -> Dict[int, int]:
    """
    Load dependencies.csv and count how many times each crate_id is depended on.
    This is a rough measure of reverse dependencies (dependency edges).
    """
    log("[sample_crates] Reading dependencies.csv (for reverse deps) ...")
    member = find_member(tf, "dependencies.csv")
    revdeps: Dict[int, int] = defaultdict(int)
    with tf.extractfile(member) as f:
        reader = csv.DictReader((line.decode("utf-8") for line in f))
        for row in reader:
            try:
                cid = int(row["crate_id"])
            except (KeyError, ValueError):
                continue
            revdeps[cid] += 1
    return revdeps


def load_versions_latest_year(tf: tarfile.TarFile) -> Dict[int, Tuple[str, Optional[int]]]:
    """
    Load versions.csv and compute, for each crate_id:
      - latest_version_created_at (string)
      - latest_version_year (int or None)
    We use max(created_at) as a proxy for "recent activity".
    """
    log("[sample_crates] Scanning versions.csv for latest version dates ...")
    member = find_member(tf, "versions.csv")
    latest_created: Dict[int, str] = {}
    with tf.extractfile(member) as f:
        reader = csv.DictReader((line.decode("utf-8") for line in f))
        for row in reader:
            try:
                cid = int(row["crate_id"])
            except (KeyError, ValueError):
                continue
            created_at = row.get("created_at", "") or ""
            if not created_at:
                continue
            # Simple string comparison works for ISO-like timestamps
            prev = latest_created.get(cid)
            if prev is None or created_at > prev:
                latest_created[cid] = created_at

    result: Dict[int, Tuple[str, Optional[int]]] = {}
    for cid, ts in latest_created.items():
        year = None
        if len(ts) >= 4 and ts[:4].isdigit():
            year = int(ts[:4])
        result[cid] = (ts, year)
    return result


def load_categories(tf: tarfile.TarFile) -> Tuple[Dict[int, str], Dict[int, List[int]]]:
    """
    Load categories.csv and crates_categories.csv:

    Returns:
      - categories: {category_id: slug}
      - crate_cats: {crate_id: [category_id, ...]}
    """
    log("[sample_crates] Reading categories.csv / crates_categories.csv ...")

    # categories.csv
    cat_member = find_member(tf, "categories.csv")
    categories: Dict[int, str] = {}
    with tf.extractfile(cat_member) as f:
        reader = csv.DictReader((line.decode("utf-8") for line in f))
        for row in reader:
            try:
                cid = int(row["id"])
            except (KeyError, ValueError):
                continue
            slug = row.get("slug", "") or ""
            categories[cid] = slug

    # crates_categories.csv
    cc_member = find_member(tf, "crates_categories.csv")
    crate_cats: Dict[int, List[int]] = defaultdict(list)
    with tf.extractfile(cc_member) as f:
        reader = csv.DictReader((line.decode("utf-8") for line in f))
        for row in reader:
            try:
                crate_id = int(row["crate_id"])
                category_id = int(row["category_id"])
            except (KeyError, ValueError):
                continue
            crate_cats[crate_id].append(category_id)

    return categories, crate_cats


def get_top_category(crate_id: int, categories: Dict[int, str], crate_cats: Dict[int, List[int]]) -> str:
    """
    Choose a primary top-level category for a crate:
      - If it has categories, pick the first category_id (arbitrary but stable),
        take its slug, and then slug.split("::", 1)[0] as the top-level.
      - If no categories, return "uncategorized".
    """
    cat_ids = crate_cats.get(crate_id)
    if not cat_ids:
        return "uncategorized"
    # Arbitrary but deterministic: pick the first one
    first_cid = cat_ids[0]
    slug = categories.get(first_cid, "") or ""
    if not slug:
        return "uncategorized"
    return slug.split("::", 1)[0]


def quantiles(sorted_values: List[int], ps: List[float]) -> Dict[float, float]:
    """
    Compute empirical quantiles on sorted list.
    """
    n = len(sorted_values)
    if n == 0:
        return {p: float("nan") for p in ps}
    result: Dict[float, float] = {}
    for p in ps:
        if p <= 0:
            result[p] = float(sorted_values[0])
        elif p >= 1:
            result[p] = float(sorted_values[-1])
        else:
            idx = p * (n - 1)
            lo = math.floor(idx)
            hi = math.ceil(idx)
            if lo == hi:
                result[p] = float(sorted_values[lo])
            else:
                w = idx - lo
                result[p] = sorted_values[lo] * (1 - w) + sorted_values[hi] * w
    return result


def format_int(n: int) -> str:
    return f"{n:,}"


def load_core_list(path: str) -> set:
    """
    Load a CSV with at least a 'name' column of crate names.
    Ignoring other columns.
    """
    core_names = set()
    with open(path, "r", encoding="utf-8") as f:
        # Try to detect delimiter quickly
        first_line = f.readline()
        if not first_line:
            return core_names
        delim = "\t" if "\t" in first_line and "," not in first_line else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        if "name" not in reader.fieldnames:
            raise ValueError(f"--core-list {path} must have a 'name' column")
        for row in reader:
            name = (row.get("name") or "").strip()
            if name:
                core_names.add(name)
    return core_names


def allocate_strata_sample_sizes(
    strata_counts: Dict[Tuple[str, str], int],
    target: int,
) -> Dict[Tuple[str, str], int]:
    """
    Given population counts per stratum and a total target size,
    compute how many samples to take from each stratum, proportional to their size.

    Uses floor(quota) + distributing leftover by largest fractional parts.
    """
    total = sum(strata_counts.values())
    if total == 0 or target <= 0:
        return {s: 0 for s in strata_counts}

    quotas: Dict[Tuple[str, str], float] = {}
    base_counts: Dict[Tuple[str, str], int] = {}
    remainders: List[Tuple[float, Tuple[str, str]]] = []

    for s, n in strata_counts.items():
        q = target * (n / total)
        quotas[s] = q
        b = int(math.floor(q))
        base_counts[s] = min(b, n)  # cannot exceed population
        remainders.append((q - b, s))

    assigned = sum(base_counts.values())
    leftover = target - assigned

    # Distribute leftover to strata with largest fractional parts,
    # while respecting population size.
    remainders.sort(reverse=True, key=lambda x: x[0])

    i = 0
    while leftover > 0 and i < len(remainders):
        frac, s = remainders[i]
        if frac <= 0:
            break
        if base_counts[s] < strata_counts[s]:
            base_counts[s] += 1
            leftover -= 1
        i += 1
        if i == len(remainders) and leftover > 0:
            # Loop again if we still have leftover; this is rare.
            i = 0

    # Final sanity: cap at population
    for s, n in strata_counts.items():
        if base_counts[s] > n:
            base_counts[s] = n

    return base_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="Path to crates.io db-dump.tar.gz")
    ap.add_argument("--out", required=True, help="Output CSV for sampled crates")
    ap.add_argument(
        "--target-size",
        type=int,
        default=1500,
        help="Total desired sample size (core + non-core), default 1500",
    )
    ap.add_argument(
        "--core-top-revdeps",
        type=int,
        default=300,
        help="Number of top crates by reverse deps to include in core stratum",
    )
    ap.add_argument(
        "--core-list",
        help="Optional CSV with a 'name' column listing extra core crates (e.g., heavy nightly users)",
    )
    ap.add_argument(
        "--min-downloads-noncore",
        type=int,
        default=100,
        help="Minimum total downloads for non-core candidates (default 100)",
    )
    ap.add_argument(
        "--min-latest-year-noncore",
        type=int,
        default=2015,
        help="Minimum latest version year for non-core candidates (default 2015)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for non-core sampling (default 42)",
    )

    args = ap.parse_args()

    log("[sample_crates] Starting two-layer sampling pipeline ...")
    log(f"[sample_crates] Opening DB dump: {args.dump}")
    tf = tarfile.open(args.dump, "r:gz")

    # Load base data
    crates = load_crates(tf)
    downloads = load_downloads(tf)
    revdeps = load_reverse_deps(tf)
    latest_map = load_versions_latest_year(tf)
    categories, crate_cats = load_categories(tf)

    # Build metrics for all crates
    log("[sample_crates] Building crate metrics ...")
    all_meta: List[CrateMeta] = []
    for cid, cinfo in crates.items():
        name = cinfo["name"]
        created_at = cinfo.get("created_at", "") or ""
        repo = cinfo.get("repository", "") or ""
        dl = downloads.get(cid, 0)
        rd = revdeps.get(cid, 0)
        latest_created_at, latest_year = latest_map.get(cid, ("", None))
        top_cat = get_top_category(cid, categories, crate_cats)
        all_meta.append(
            CrateMeta(
                crate_id=cid,
                name=name,
                repository=repo,
                created_at=created_at,
                latest_version_created_at=latest_created_at,
                latest_version_year=latest_year,
                downloads=dl,
                revdeps=rd,
                top_category=top_cat,
            )
        )

    total_crates = len(all_meta)
    log(f"[sample_crates] Total crates in dump: {format_int(total_crates)}")

    # --- Core stratum: top by reverse deps + optional core list ---

    # Sort by revdeps descending
    sorted_by_revdeps = sorted(all_meta, key=lambda m: m.revdeps, reverse=True)
    core_by_revdeps = [
        m for m in sorted_by_revdeps if m.revdeps > 0
    ][: args.core_top_revdeps]

    core_ids = {m.crate_id for m in core_by_revdeps}
    core_names_set = {m.name for m in core_by_revdeps}

    if args.core_list:
        log(f"[sample_crates] Loading extra core names from {args.core_list} ...")
        extra_names = load_core_list(args.core_list)
        core_names_set |= extra_names

        # Expand core_ids based on names
        name_to_meta = {m.name: m for m in all_meta}
        for nm in extra_names:
            m = name_to_meta.get(nm)
            if m:
                core_ids.add(m.crate_id)

    core_meta = [m for m in all_meta if m.crate_id in core_ids]
    core_size = len(core_meta)
    log(f"[sample_crates] Core stratum size: {format_int(core_size)} crates")

    # --- Non-core candidate pool ---

    rnd = random.Random(args.seed)

    noncore_all = [m for m in all_meta if m.crate_id not in core_ids]

    # Apply filters: min downloads and recency
    noncore_candidates: List[CrateMeta] = []
    for m in noncore_all:
        if m.downloads < args.min_downloads_noncore:
            continue
        if m.latest_version_year is not None and m.latest_version_year < args.min_latest_year_noncore:
            continue
        # If latest_version_year is None (we didn't see versions), we keep it for now.
        noncore_candidates.append(m)

    nc_total = len(noncore_candidates)
    log(
        f"[sample_crates] Non-core candidates after filters "
        f"(downloads >= {args.min_downloads_noncore}, latest_year >= {args.min_latest_year_noncore} if known): "
        f"{format_int(nc_total)} crates"
    )

    if nc_total == 0:
        log("[sample_crates] WARNING: No non-core candidates after filters. Output will contain only core crates.")

    # If target size is smaller than core size, we just output all cores.
    if args.target_size <= core_size:
        log(
            "[sample_crates] Target size <= core size; "
            "output will contain only core stratum (no non-core sample)."
        )
        sampled = core_meta
        # Write output and exit
        with open(args.out, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(
                [
                    "crate_id",
                    "name",
                    "repository",
                    "downloads",
                    "revdeps",
                    "top_category",
                    "popularity_band",
                    "is_core",
                    "stratum",
                    "created_at",
                    "latest_version_created_at",
                    "latest_version_year",
                ]
            )
            for m in sampled:
                writer.writerow(
                    [
                        m.crate_id,
                        m.name,
                        m.repository,
                        m.downloads,
                        m.revdeps,
                        m.top_category,
                        "n/a",
                        1,
                        "core",
                        m.created_at,
                        m.latest_version_created_at,
                        m.latest_version_year if m.latest_version_year is not None else "",
                    ]
                )
        log(f"[sample_crates] Sample CSV written to {args.out}")
        log(
            f"[sample_crates] FINAL SUMMARY: total sampled crates = {format_int(len(sampled))} "
            f"(core only)"
        )
        return

    # --- Popularity bands on non-core candidates ---

    dl_values_nc = sorted(m.downloads for m in noncore_candidates)
    q_nc = quantiles(dl_values_nc, [0.5, 0.9])
    q50_nc = q_nc[0.5]
    q90_nc = q_nc[0.9]
    log(
        "[sample_crates] Non-core downloads quantiles: "
        f"p50={format_int(int(q50_nc))}, p90={format_int(int(q90_nc))}"
    )

    def get_pop_band(dl: int) -> str:
        if dl < q50_nc:
            return "low"
        elif dl < q90_nc:
            return "mid"
        else:
            return "high"

    # Count population per stratum (top_category, pop_band)
    strata_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for m in noncore_candidates:
        band = get_pop_band(m.downloads)
        strata_counts[(m.top_category, band)] += 1

    # Determine non-core target size
    target_noncore = args.target_size - core_size
    if target_noncore > nc_total:
        log(
            f"[sample_crates] WARNING: target non-core size {target_noncore} "
            f"exceeds candidate pool {nc_total}. We will take all non-core candidates."
        )
        target_noncore = nc_total

    log(
        f"[sample_crates] Target non-core sample size: {format_int(target_noncore)} "
        f"(given total target {format_int(args.target_size)})"
    )

    # Allocate sample sizes per stratum
    strata_sample_sizes = allocate_strata_sample_sizes(strata_counts, target_noncore)

    # Build index of candidates by stratum
    strata_members: Dict[Tuple[str, str], List[CrateMeta]] = defaultdict(list)
    for m in noncore_candidates:
        band = get_pop_band(m.downloads)
        strata_members[(m.top_category, band)].append(m)

    # Sample within each stratum
    sampled_noncore: List[CrateMeta] = []
    for s, n_pop in strata_counts.items():
        n_sample = strata_sample_sizes.get(s, 0)
        if n_sample <= 0:
            continue
        members = strata_members[s]
        if n_sample >= len(members):
            # Take all
            chosen = members
        else:
            chosen = rnd.sample(members, n_sample)
        sampled_noncore.extend(chosen)

    # Sanity: cap / truncate 以防 rounding 偶尔多出一两个
    if len(sampled_noncore) > target_noncore:
        sampled_noncore = sampled_noncore[:target_noncore]

    total_sampled = core_size + len(sampled_noncore)
    log(
        f"[sample_crates] Sample sizes: core={format_int(core_size)}, "
        f"non-core={format_int(len(sampled_noncore))}, total={format_int(total_sampled)}"
    )

    # --- Write output CSV ---

    with open(args.out, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            [
                "crate_id",
                "name",
                "repository",
                "downloads",
                "revdeps",
                "top_category",
                "popularity_band",
                "is_core",
                "stratum",
                "created_at",
                "latest_version_created_at",
                "latest_version_year",
            ]
        )

        # Core first
        for m in core_meta:
            writer.writerow(
                [
                    m.crate_id,
                    m.name,
                    m.repository,
                    m.downloads,
                    m.revdeps,
                    m.top_category,
                    "n/a",
                    1,
                    "core",
                    m.created_at,
                    m.latest_version_created_at,
                    m.latest_version_year if m.latest_version_year is not None else "",
                ]
            )

        # Non-core
        for m in sampled_noncore:
            band = get_pop_band(m.downloads)
            writer.writerow(
                [
                    m.crate_id,
                    m.name,
                    m.repository,
                    m.downloads,
                    m.revdeps,
                    m.top_category,
                    band,
                    0,
                    "noncore",
                    m.created_at,
                    m.latest_version_created_at,
                    m.latest_version_year if m.latest_version_year is not None else "",
                ]
            )

    # --- Print summary info ---

    # Category summary
    cat_counts_core: Dict[str, int] = defaultdict(int)
    cat_counts_nc: Dict[str, int] = defaultdict(int)
    for m in core_meta:
        cat_counts_core[m.top_category] += 1
    for m in sampled_noncore:
        cat_counts_nc[m.top_category] += 1

    log("\n===== SAMPLING SUMMARY =====")
    log(f"Total crates in dump         : {format_int(total_crates)}")
    log(f"Core stratum size            : {format_int(core_size)}")
    log(f"Non-core candidate pool size : {format_int(nc_total)}")
    log(f"Non-core sampled             : {format_int(len(sampled_noncore))}")
    log(f"TOTAL SAMPLED                : {format_int(total_sampled)}")

    log("\nTop-level categories in core sample (top 15):")
    for cat, cnt in sorted(cat_counts_core.items(), key=lambda x: x[1], reverse=True)[:15]:
        log(f"  {cat:<25} {cnt:5d}")

    log("\nTop-level categories in non-core sample (top 15):")
    for cat, cnt in sorted(cat_counts_nc.items(), key=lambda x: x[1], reverse=True)[:15]:
        log(f"  {cat:<25} {cnt:5d}")

    log("============================")
    log(f"[sample_crates] Sample CSV written to {args.out}")
    log("[sample_crates] Done.")


if __name__ == "__main__":
    main()
