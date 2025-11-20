#!/usr/bin/env python3
import argparse
import csv
import io
import os
import random
import sys
import tarfile
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


# ---------- 小工具 ----------
# --- 关键补丁：放大 CSV 单元格大小限制 ---
# crates.csv 里有很长的字段（比如说明文档），默认 128KB 不够用，
# 我们把它调到接近系统允许的最大值。
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)
# --- 补丁到此结束 ---



@dataclass
class CrateMeta:
    crate_id: int
    name: str
    repository: str
    downloads: int
    top_category: str
    all_categories: str
    created_year: Optional[int]
    updated_year: Optional[int]
    pop_stratum: str  # "high" / "mid" / "low"


def log(msg: str):
    """简单的日志输出，方便确认脚本在运行。"""
    print(f"[sample_crates] {msg}")


def parse_year(date_str: Optional[str]) -> Optional[int]:
    """从日期字符串中取年份（YYYY...）。"""
    if not date_str:
        return None
    if len(date_str) >= 4 and date_str[:4].isdigit():
        return int(date_str[:4])
    return None


def find_member(tar: tarfile.TarFile, suffix: str) -> Optional[tarfile.TarInfo]:
    """在 tar 里找到以某个后缀结尾的成员（例如 crates.csv）。"""
    for m in tar.getmembers():
        if m.name.endswith(suffix):
            return m
    return None


def read_csv_from_tar(
    tar: tarfile.TarFile,
    suffix: str,
    required_columns: List[str],
) -> List[Dict[str, str]]:
    """从 db-dump 中读取 CSV，并检查列是否存在。"""
    member = find_member(tar, suffix)
    if member is None:
        raise RuntimeError(
            f"Could not find {suffix} inside db-dump tarball. "
            "Make sure you downloaded the official crates.io DB dump."
        )
    f = tar.extractfile(member)
    if f is None:
        raise RuntimeError(f"Could not extract {suffix} from tarball.")
    with io.TextIOWrapper(f, encoding="utf-8") as text:
        reader = csv.DictReader(text)
        if reader.fieldnames is None:
            raise RuntimeError(f"{suffix} has no header row.")
        missing = [c for c in required_columns if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(
                f"{suffix} is missing expected columns: {missing}. "
                "The DB dump format may have changed; please update the script."
            )
        rows = list(reader)
    return rows


# ---------- 载入 DB DUMP ----------

def load_metadata_from_dump(dump_path: str):
    """从 db-dump.tar.gz 里读出我们需要的几张表。"""
    if not os.path.exists(dump_path):
        raise FileNotFoundError(
            f"DB dump file {dump_path!r} not found. "
            "Download it from https://static.crates.io/db-dump.tar.gz "
            "and pass its path via --dump."
        )

    log(f"Opening DB dump: {dump_path}")
    with tarfile.open(dump_path, "r:gz") as tar:
        log("Reading crates.csv ...")
        crates_rows = read_csv_from_tar(
            tar,
            "crates.csv",
            ["id", "name", "created_at", "updated_at", "repository"],
        )

        log("Reading crate_downloads.csv ...")
        crate_downloads_rows = read_csv_from_tar(
            tar,
            "crate_downloads.csv",
            ["crate_id", "downloads"],
        )

        log("Reading categories.csv / crates_categories.csv ...")
        categories_rows = read_csv_from_tar(
            tar,
            "categories.csv",
            ["id", "slug"],
        )
        crates_categories_rows = read_csv_from_tar(
            tar,
            "crates_categories.csv",
            ["crate_id", "category_id"],
        )

        log("Reading default_versions.csv ...")
        default_versions_rows = read_csv_from_tar(
            tar,
            "default_versions.csv",
            ["crate_id", "version_id", "num_versions"],
        )

        log("Scanning versions.csv for default versions ...")
        default_version_ids = {int(row["version_id"]) for row in default_versions_rows}
        versions_member = find_member(tar, "versions.csv")
        if versions_member is None:
            raise RuntimeError("Could not find versions.csv in DB dump.")
        version_meta_by_id: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        vf = tar.extractfile(versions_member)
        if vf is None:
            raise RuntimeError("Could not extract versions.csv from DB dump.")
        with io.TextIOWrapper(vf, encoding="utf-8") as vtext:
            vreader = csv.DictReader(vtext)
            if vreader.fieldnames is None:
                raise RuntimeError("versions.csv has no header row.")
            missing = [c for c in ["id", "created_at", "updated_at"] if c not in vreader.fieldnames]
            if missing:
                raise RuntimeError(
                    f"versions.csv is missing expected columns: {missing}. "
                    "The DB dump format may have changed; please update the script."
                )
            for i, row in enumerate(vreader):
                try:
                    vid = int(row["id"])
                except ValueError:
                    continue
                if vid not in default_version_ids:
                    continue
                c_year = parse_year(row.get("created_at"))
                u_year = parse_year(row.get("updated_at"))
                version_meta_by_id[vid] = (c_year, u_year)
                # 可选：每隔几十万行打个点，这里先省略

    log("Building in-memory indexes ...")

    crates_by_id: Dict[int, Dict[str, str]] = {}
    for row in crates_rows:
        try:
            cid = int(row["id"])
        except ValueError:
            continue
        crates_by_id[cid] = row

    downloads_by_crate_id: Dict[int, int] = defaultdict(int)
    for row in crate_downloads_rows:
        try:
            cid = int(row["crate_id"])
            dl = int(row["downloads"])
        except ValueError:
            continue
        downloads_by_crate_id[cid] += dl

    categories_by_id: Dict[int, str] = {}
    for row in categories_rows:
        try:
            cat_id = int(row["id"])
        except ValueError:
            continue
        slug = row.get("slug") or ""
        categories_by_id[cat_id] = slug

    crate_to_categories: Dict[int, List[str]] = defaultdict(list)
    for row in crates_categories_rows:
        try:
            cid = int(row["crate_id"])
            cat_id = int(row["category_id"])
        except ValueError:
            continue
        slug = categories_by_id.get(cat_id)
        if slug:
            crate_to_categories[cid].append(slug)

    crate_to_default_version_id: Dict[int, int] = {}
    for row in default_versions_rows:
        try:
            cid = int(row["crate_id"])
            vid = int(row["version_id"])
        except ValueError:
            continue
        crate_to_default_version_id[cid] = vid

    log("Metadata loaded.")
    return (
        crates_by_id,
        downloads_by_crate_id,
        crate_to_categories,
        crate_to_default_version_id,
        version_meta_by_id,
    )


# ---------- 构建候选集合 ----------

def build_candidate_crates(
    crates_by_id,
    downloads_by_crate_id,
    crate_to_categories,
    crate_to_default_version_id,
    version_meta_by_id,
    min_year: int,
    min_downloads: int,
    require_github: bool = True,
):
    """根据过滤条件构建候选 crate 列表。"""
    log(f"Building candidate set with min_year={min_year}, min_downloads={min_downloads} ...")
    candidates: List[CrateMeta] = []

    for cid, crate_row in crates_by_id.items():
        name = crate_row.get("name") or ""
        repo = (crate_row.get("repository") or "").strip()

        # 1) 只要 GitHub 仓库
        if require_github:
            repo_lower = repo.lower()
            if "github.com" not in repo_lower:
                continue

        # 2) 下载量过滤
        downloads = downloads_by_crate_id.get(cid, 0)
        if downloads < min_downloads:
            continue

        # 3) 时间过滤：优先用 default version 的 created/updated 年份
        created_year = None
        updated_year = None
        vid = crate_to_default_version_id.get(cid)
        if vid is not None and vid in version_meta_by_id:
            created_year, updated_year = version_meta_by_id[vid]
        if created_year is None:
            created_year = parse_year(crate_row.get("created_at"))
        if updated_year is None:
            updated_year = parse_year(crate_row.get("updated_at"))

        year_for_filter = updated_year or created_year
        if year_for_filter is None or year_for_filter < min_year:
            continue

        # 4) 分类：取一个“顶级 category”（slug 的前半段）
        cats = crate_to_categories.get(cid, [])
        if cats:
            top_cats = sorted({c.split("::", 1)[0] for c in cats})
            top_category = top_cats[0]
            all_categories = ",".join(sorted(cats))
        else:
            top_category = "uncategorized"
            all_categories = ""

        candidates.append(
            CrateMeta(
                crate_id=cid,
                name=name,
                repository=repo,
                downloads=downloads,
                top_category=top_category,
                all_categories=all_categories,
                created_year=created_year,
                updated_year=updated_year,
                pop_stratum="",
            )
        )

    log(f"Candidate crates after filters: {len(candidates)}")
    return candidates


def assign_pop_strata(candidates: List[CrateMeta]):
    """给每个候选 crate 按下载量打 high / mid / low 标签。"""
    if not candidates:
        return

    sorted_dl = sorted(c.downloads for c in candidates)
    n = len(sorted_dl)

    def q(p: float) -> int:
        if n == 1:
            return sorted_dl[0]
        idx = int(p * (n - 1))
        return sorted_dl[idx]

    q50 = q(0.5)
    q80 = q(0.8)
    log(f"Download quantiles: q50={q50}, q80={q80}")

    for c in candidates:
        if c.downloads >= q80:
            c.pop_stratum = "high"
        elif c.downloads >= q50:
            c.pop_stratum = "mid"
        else:
            c.pop_stratum = "low"


def stratified_sample(candidates: List[CrateMeta], target_size: int, seed: int) -> List[CrateMeta]:
    """按 (top_category, pop_stratum) 分层抽样。"""
    random.seed(seed)
    if len(candidates) <= target_size:
        log(f"Candidates ({len(candidates)}) <= target_size ({target_size}), no sampling needed.")
        return list(candidates)

    strata: Dict[Tuple[str, str], List[CrateMeta]] = defaultdict(list)
    for c in candidates:
        key = (c.top_category, c.pop_stratum)
        strata[key].append(c)

    total = len(candidates)
    log(f"Strata count: {len(strata)}")
    sampled: List[CrateMeta] = []

    for key, items in strata.items():
        frac = len(items) / total
        k = int(round(frac * target_size))
        if k < 3 and len(items) >= 3:
            k = 3
        if k > len(items):
            k = len(items)
        if k == 0:
            continue
        chosen = random.sample(items, k)
        sampled.extend(chosen)

    if len(sampled) > target_size:
        log(f"Oversampled {len(sampled)} > {target_size}, trimming down.")
        sampled = random.sample(sampled, target_size)
    elif len(sampled) < target_size:
        need = target_size - len(sampled)
        log(f"Sampled {len(sampled)} < {target_size}, filling with extra {need}.")
        remaining = [c for c in candidates if c not in sampled]
        need = min(need, len(remaining))
        sampled.extend(random.sample(remaining, need))

    return sampled


def write_sample_csv(sampled: List[CrateMeta], out_path: str):
    """把抽样结果写成 CSV。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = [
        "crate_id",
        "name",
        "repository",
        "downloads",
        "top_category",
        "all_categories",
        "created_year",
        "updated_year",
        "pop_stratum",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in sampled:
            writer.writerow(asdict(c))
    log(f"Sample CSV written to {out_path}")


def print_summary(candidates: List[CrateMeta], sampled: List[CrateMeta]):
    """在控制台上打印摘要。"""
    print()
    print("===== SAMPLING SUMMARY =====")
    print(f"Total candidate crates after filters: {len(candidates)}")
    print(f"Sample size: {len(sampled)}")

    cat_counts = Counter(c.top_category for c in sampled)
    print("\nTop-level categories in sample (top 15):")
    for cat, count in cat_counts.most_common(15):
        print(f"  {cat:25s} {count:4d}")

    pop_counts = Counter(c.pop_stratum for c in sampled)
    print("\nPopularity strata in sample:")
    for s, count in sorted(pop_counts.items()):
        print(f"  {s:6s}: {count}")
    print("============================")


# ---------- 主入口 ----------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stratified sample of crates.io crates (GitHub-backed) from DB dump."
    )
    parser.add_argument(
        "--dump",
        required=True,
        help="Path to crates.io DB dump tar.gz "
             "(download from https://static.crates.io/db-dump.tar.gz)",
    )
    parser.add_argument(
        "--out",
        default="sampled_crates.csv",
        help="Output CSV path (default: sampled_crates.csv)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=700,
        help="Target sample size (default: 700)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2020,
        help="Minimum activity year (default: 2020)",
    )
    parser.add_argument(
        "--min-downloads",
        type=int,
        default=100,
        help="Minimum total downloads (default: 100)",
    )
    args = parser.parse_args(argv)

    log("Starting sampling pipeline ...")

    try:
        (
            crates_by_id,
            downloads_by_crate_id,
            crate_to_categories,
            crate_to_default_version_id,
            version_meta_by_id,
        ) = load_metadata_from_dump(args.dump)
    except Exception as e:
        print("FATAL: could not load metadata from DB dump.", file=sys.stderr)
        print(f"Reason: {e}", file=sys.stderr)
        print(
            "\nHow to fix:\n"
            "1. Ensure you downloaded the official DB dump:\n"
            "   curl -L -o db-dump.tar.gz https://static.crates.io/db-dump.tar.gz\n"
            "2. Re-run this script with:  --dump ./db-dump.tar.gz\n"
            "3. If the error persists, inspect the CSVs inside the tarball "
            "   and update column names in the script.",
            file=sys.stderr,
        )
        sys.exit(1)

    candidates = build_candidate_crates(
        crates_by_id=crates_by_id,
        downloads_by_crate_id=downloads_by_crate_id,
        crate_to_categories=crate_to_categories,
        crate_to_default_version_id=crate_to_default_version_id,
        version_meta_by_id=version_meta_by_id,
        min_year=args.min_year,
        min_downloads=args.min_downloads,
    )

    if not candidates:
        print(
            "No candidate crates after applying filters.\n"
            "Try lowering --min-downloads or --min-year.",
            file=sys.stderr,
        )
        sys.exit(1)

    assign_pop_strata(candidates)
    sampled = stratified_sample(candidates, args.target_size, args.seed)
    write_sample_csv(sampled, args.out)
    print_summary(candidates, sampled)
    log("Done.")


if __name__ == "__main__":
    main()
