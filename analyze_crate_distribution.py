#!/usr/bin/env python3
"""
analyze_crate_distribution.py

Usage:
  python3 analyze_crate_distribution.py \
    --dump ./db-dump.tar.gz \
    --core-list ./core_nightly.csv

- --dump: crates.io db dump tar.gz
- --core-list: (optional) CSV with at least a 'name' column of crate names
"""

import argparse
import csv
import math
import tarfile
from collections import defaultdict
from statistics import median

# 避免 crates.csv 里 description 太长触发 field limit
csv.field_size_limit(2**31 - 1)


def find_member(tf: tarfile.TarFile, basename: str) -> tarfile.TarInfo:
    """
    在 tar 里按文件名结尾查找，比如 'crates.csv', 'crate_downloads.csv'。
    有些 dump 在根目录，有些在 data/ 下面，这里统一做一层模糊匹配。
    """
    for m in tf.getmembers():
        if m.name.endswith("/" + basename) or m.name == basename:
            return m
    raise FileNotFoundError(f"Could not find {basename} in tarball")


def load_crates_and_downloads(tf: tarfile.TarFile):
    """
    读取 crates.csv 和 crate_downloads.csv，返回：
      - crates: {crate_id: {"name": str, "created_at": str}}
      - downloads: {crate_id: total_downloads}
    """
    # 1) crates.csv
    crates = {}
    crates_member = find_member(tf, "crates.csv")
    with tf.extractfile(crates_member) as f:
        reader = csv.DictReader((line.decode("utf-8") for line in f))
        for row in reader:
            cid = int(row["id"])
            name = row["name"]
            created_at = row.get("created_at", "")
            crates[cid] = {
                "name": name,
                "created_at": created_at,
            }

    # 2) crate_downloads.csv
    downloads = defaultdict(int)
    cd_member = find_member(tf, "crate_downloads.csv")
    with tf.extractfile(cd_member) as f:
        reader = csv.DictReader((line.decode("utf-8") for line in f))
        for row in reader:
            cid = int(row["crate_id"])
            d = int(row["downloads"])
            downloads[cid] += d

    return crates, downloads


def load_reverse_deps(tf: tarfile.TarFile):
    """
    从 dependencies.csv 统计每个 crate 被依赖的次数（粗略的 reverse deps）。
    crates.io 的 dependencies.csv 通常有字段：
      id, version_id, crate_id, req, optional, default_features, features,
      target, kind, explicit_name
    这里 crate_id 是“被依赖的 crate”的 id，
    每出现一行就说明“某个版本依赖了这个 crate 一次”。

    返回: {crate_id: reverse_dep_count}
    """
    revdeps = defaultdict(int)
    dep_member = find_member(tf, "dependencies.csv")
    with tf.extractfile(dep_member) as f:
        reader = csv.DictReader((line.decode("utf-8") for line in f))
        for row in reader:
            try:
                cid = int(row["crate_id"])
            except (KeyError, ValueError):
                # 如果字段名对不上，请手动检查 dependencies.csv header
                continue
            revdeps[cid] += 1
    return revdeps


def quantiles(sorted_values, ps):
    """
    计算一组分位数，sorted_values 需已升序。
    ps: [0.5, 0.9, ...] 这样的 list
    """
    n = len(sorted_values)
    if n == 0:
        return {p: None for p in ps}
    result = {}
    for p in ps:
        if p <= 0:
            result[p] = sorted_values[0]
        elif p >= 1:
            result[p] = sorted_values[-1]
        else:
            idx = p * (n - 1)
            lo = math.floor(idx)
            hi = math.ceil(idx)
            if lo == hi:
                result[p] = sorted_values[lo]
            else:
                w = idx - lo
                result[p] = sorted_values[lo] * (1 - w) + sorted_values[hi] * w
    return result


def format_int(n):
    return f"{n:,}"


def format_ratio(x, y):
    if y == 0:
        return "n/a"
    return f"{x/y:.4f}"


def load_core_list(path: str):
    """
    读取你那份 heavy-nightly 列表，要求至少有 name 列。
    允许多余字段（url, github_stars, downloads...）。
    返回 set(core_crate_names)
    """
    core_names = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t" if "\t" in f.readline() else ",")
        f.seek(0)
        reader = csv.DictReader(f)
        if "name" not in reader.fieldnames:
            raise ValueError(f"--core-list {path} must have a 'name' column")
        for row in reader:
            name = row["name"].strip()
            if name:
                core_names.add(name)
    return core_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="Path to crates.io db-dump.tar.gz")
    ap.add_argument(
        "--core-list",
        help="Optional CSV with a 'name' column of core/heavy-nightly crates",
    )
    args = ap.parse_args()

    print(f"[+] Opening dump: {args.dump}")
    with tarfile.open(args.dump, "r:gz") as tf:
        crates, downloads = load_crates_and_downloads(tf)
        revdeps = load_reverse_deps(tf)

    # 组装 metrics
    metrics = []
    for cid, info in crates.items():
        name = info["name"]
        total_dl = downloads.get(cid, 0)
        rd = revdeps.get(cid, 0)
        created_at = info["created_at"]
        metrics.append(
            {
                "id": cid,
                "name": name,
                "downloads": total_dl,
                "revdeps": rd,
                "created_at": created_at,
            }
        )

    total_crates = len(metrics)
    print(f"[+] Total crates in dump: {total_crates:,}")

    # 下载量分布
    dl_values = sorted(m["downloads"] for m in metrics)
    total_downloads = sum(dl_values)

    print("\n=== Downloads distribution ===")
    print(f"Total downloads over all crates: {format_int(total_downloads)}")

    dl_q = quantiles(dl_values, [0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    for p, v in dl_q.items():
        print(f"  p{int(p*100):3d} = {format_int(int(v))}")

    # 各种阈值下的 crate 数量
    thresholds = [0, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    print("\nCrates count by downloads threshold:")
    for t in thresholds:
        cnt = sum(1 for m in metrics if m["downloads"] >= t)
        print(
            f"  downloads >= {format_int(t):>8}: {cnt:>7,} crates "
            f"({cnt/total_crates*100:5.2f}%)"
        )

    # top-K 覆盖比例
    metrics_sorted_by_dl = sorted(metrics, key=lambda m: m["downloads"], reverse=True)
    print("\nContribution of top-K crates by downloads:")
    for K in [10, 50, 100, 500, 1000]:
        if K > total_crates:
            break
        s = sum(m["downloads"] for m in metrics_sorted_by_dl[:K])
        print(
            f"  top {K:4d}: {format_int(s)} downloads "
            f"({s/total_downloads*100:6.2f}% of all downloads)"
        )

    print("\nTop 20 crates by downloads:")
    for i, m in enumerate(metrics_sorted_by_dl[:20], start=1):
        print(
            f"  #{i:2d} {m['name']:<30} "
            f"downloads={format_int(m['downloads'])} "
            f"revdeps={format_int(m['revdeps'])}"
        )

    # 反向依赖分布
    rd_values = sorted(m["revdeps"] for m in metrics)
    total_revdeps = sum(rd_values)

    print("\n=== Reverse dependency distribution ===")
    print(f"Total dependency edges (approx): {format_int(total_revdeps)}")

    rd_q = quantiles(rd_values, [0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    for p, v in rd_q.items():
        print(f"  p{int(p*100):3d} = {format_int(int(v))}")

    # 阈值统计
    rd_thresholds = [0, 1, 5, 10, 50, 100, 500, 1000]
    print("\nCrates count by reverse deps threshold:")
    for t in rd_thresholds:
        cnt = sum(1 for m in metrics if m["revdeps"] >= t)
        print(
            f"  revdeps >= {format_int(t):>4}: {cnt:>7,} crates "
            f"({cnt/total_crates*100:5.2f}%)"
        )

    metrics_sorted_by_rd = sorted(metrics, key=lambda m: m["revdeps"], reverse=True)
    print("\nContribution of top-K crates by reverse deps:")
    for K in [10, 50, 100, 500, 1000]:
        if K > total_crates:
            break
        s = sum(m["revdeps"] for m in metrics_sorted_by_rd[:K])
        print(
            f"  top {K:4d}: {format_int(s)} dependency edges "
            f"({format_ratio(s, total_revdeps)} of all edges)"
        )

    print("\nTop 20 crates by reverse deps:")
    for i, m in enumerate(metrics_sorted_by_rd[:20], start=1):
        print(
            f"  #{i:2d} {m['name']:<30} "
            f"revdeps={format_int(m['revdeps'])} "
            f"downloads={format_int(m['downloads'])}"
        )

    # 如果有 core-list，额外分析一下
    if args.core_list:
        print(f"\n[+] Loading core list from {args.core_list} ...")
        core_names = load_core_list(args.core_list)

        # 建立 name -> metrics 映射 & 排名
        name_to_metrics = {m["name"]: m for m in metrics}

        # 构建 name -> rank
        dl_rank = {m["name"]: i + 1 for i, m in enumerate(metrics_sorted_by_dl)}
        rd_rank = {m["name"]: i + 1 for i, m in enumerate(metrics_sorted_by_rd)}

        print("\n=== Core crate positions in global distributions ===")
        missing = []
        for name in sorted(core_names):
            m = name_to_metrics.get(name)
            if not m:
                missing.append(name)
                continue
            r_dl = dl_rank.get(name)
            r_rd = rd_rank.get(name)

            # 分位数估计（下载量、反向依赖）
            # rank 1 是最大值，对应分位 ~1.0
            dl_percentile = 1.0 - (r_dl - 1) / max(total_crates - 1, 1)
            rd_percentile = 1.0 - (r_rd - 1) / max(total_crates - 1, 1)

            print(
                f"  {name:<30} "
                f"downloads={format_int(m['downloads'])} "
                f"(rank {r_dl}/{total_crates}, p≈{dl_percentile*100:5.2f}%)  "
                f"revdeps={format_int(m['revdeps'])} "
                f"(rank {r_rd}/{total_crates}, p≈{rd_percentile*100:5.2f}%)"
            )

        if missing:
            print("\n[!] Core crates not found in crates.csv (name mismatch?):")
            for name in missing:
                print(f"  - {name}")


if __name__ == "__main__":
    main()
