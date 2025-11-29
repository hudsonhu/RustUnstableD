#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import threading
import time

# --------- 工具与数据结构 ---------

@dataclass
class RepoJob:
    key: str            # "owner/repo"
    owner: str
    repo: str
    git_url: str        # 统一用 https://github.com/owner/repo.git
    local_path: str     # 本地 clone 目录
    crate_names: List[str]
    top_categories: List[str]
    pop_strata: List[str]


@dataclass
class RepoStatus:
    key: str
    owner: str
    repo: str
    crate_names: str
    status: str         # "pending" / "success" / "failed"
    attempts: int
    last_error: str
    last_attempt_ts: str


def log(msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


# --------- GitHub URL 解析 ---------

def parse_github_repo(repository: str) -> Optional[Tuple[str, str, str]]:
    """
    把各种形态的 GitHub URL 解析成 (owner, repo, normalized_git_url)
    - 支持：
      - https://github.com/owner/repo.git
      - https://github.com/owner/repo
      - git@github.com:owner/repo.git
      - github.com/owner/repo
    返回 None 表示不是 GitHub 仓库或格式太奇怪。
    """
    if not repository:
        return None
    s = repository.strip()

    # SSH 形式 git@github.com:owner/repo.git
    if s.startswith("git@github.com:"):
        tail = s[len("git@github.com:"):]
        tail = tail.split("#", 1)[0].split("?", 1)[0]
        if tail.endswith(".git"):
            tail = tail[:-4]
        parts = tail.strip("/").split("/")
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1]
        normalized = f"https://github.com/{owner}/{repo}.git"
        return owner, repo, normalized

    lower = s.lower()
    if "github.com" not in lower:
        return None

    if not (s.startswith("http://") or s.startswith("https://")
            or s.startswith("git://") or s.startswith("ssh://")):
        # 例如 "github.com/owner/repo"
        s = "https://" + s.lstrip("/")

    parsed = urlparse(s)
    host = parsed.netloc.lower()
    path = parsed.path

    if "github.com" not in host:
        # 可能是 "github.com/owner/repo" 这种被当成 path 了
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

    normalized = f"https://github.com/{owner}/{repo}.git"
    return owner, repo, normalized


# --------- 读取 sampled_crates.csv 并构建 RepoJob ---------

def load_jobs_from_sample(csv_path: str, out_dir: str) -> Dict[str, RepoJob]:
    """
    从 sampled_crates*.csv 读取数据，根据 repository 字段解析 GitHub 仓库，
    聚合成 RepoJob（一个 repo 可能对应多个 crate）。

    兼容两种列名：
      - 旧版: pop_stratum
      - 新版: popularity_band
    """
    jobs: Dict[str, RepoJob] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # 基础必需列
        required_cols = ["name", "repository", "top_category"]
        for c in required_cols:
            if c not in fieldnames:
                raise RuntimeError(f"sample CSV missing column: {c}")

        # 兼容旧字段名 pop_stratum 和 新字段名 popularity_band
        pop_field = None
        if "pop_stratum" in fieldnames:
            pop_field = "pop_stratum"
        elif "popularity_band" in fieldnames:
            pop_field = "popularity_band"
        else:
            log("[WARN] sample CSV has no 'pop_stratum' or 'popularity_band'; "
                "will treat all as 'unknown'.")

        for row in reader:
            crate_name = (row.get("name") or "").strip()
            repo_url = (row.get("repository") or "").strip()
            top_category = (row.get("top_category") or "").strip() or "unknown"

            if pop_field is not None:
                pop_stratum = (row.get(pop_field) or "").strip() or "unknown"
            else:
                pop_stratum = "unknown"

            parsed = parse_github_repo(repo_url)
            if parsed is None:
                continue
            owner, repo, git_url = parsed
            key = f"{owner}/{repo}"
            local_path = os.path.join(out_dir, owner, repo)

            if key not in jobs:
                jobs[key] = RepoJob(
                    key=key,
                    owner=owner,
                    repo=repo,
                    git_url=git_url,
                    local_path=local_path,
                    crate_names=[crate_name] if crate_name else [],
                    top_categories=[top_category],
                    pop_strata=[pop_stratum],
                )
            else:
                job = jobs[key]
                if crate_name and crate_name not in job.crate_names:
                    job.crate_names.append(crate_name)
                if top_category not in job.top_categories:
                    job.top_categories.append(top_category)
                if pop_stratum not in job.pop_strata:
                    job.pop_strata.append(pop_stratum)

    return jobs


# --------- 状态文件读写（中断恢复） ---------

def load_status(status_path: str, jobs: Dict[str, RepoJob]) -> Dict[str, RepoStatus]:
    """
    从 status CSV 中读已有状态。如果文件不存在，全部标记为 pending。
    """
    status: Dict[str, RepoStatus] = {}

    if os.path.exists(status_path):
        with open(status_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get("key") or ""
                if not key:
                    continue
                status[key] = RepoStatus(
                    key=key,
                    owner=row.get("owner") or "",
                    repo=row.get("repo") or "",
                    crate_names=row.get("crate_names") or "",
                    status=row.get("status") or "pending",
                    attempts=int(row.get("attempts") or 0),
                    last_error=row.get("last_error") or "",
                    last_attempt_ts=row.get("last_attempt_ts") or "",
                )

    # 为所有 jobs 补齐没有记录的默认状态
    for key, job in jobs.items():
        if key not in status:
            status[key] = RepoStatus(
                key=key,
                owner=job.owner,
                repo=job.repo,
                crate_names=",".join(job.crate_names),
                status="pending",
                attempts=0,
                last_error="",
                last_attempt_ts="",
            )

    return status


def save_status(status_path: str, status: Dict[str, RepoStatus]):
    """
    把 status 写回 CSV。每次写全量，顺带把 last_error 里的换行清理掉。
    """
    tmp_path = status_path + ".tmp"
    fieldnames = [
        "key",
        "owner",
        "repo",
        "crate_names",
        "status",
        "attempts",
        "last_error",
        "last_attempt_ts",
    ]
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(status.keys()):
            st = status[key]
            row = asdict(st)
            le = row.get("last_error") or ""
            if le:
                le = le.replace("\r", " ").replace("\n", " | ")
                if len(le) > 300:
                    le = le[-300:]
                row["last_error"] = le
            writer.writerow(row)
    os.replace(tmp_path, status_path)


# --------- Git 操作 ---------

def run_git(args: List[str], cwd: Optional[str] = None, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    """
    运行 git 命令，返回 (returncode, stdout, stderr)。
    """
    cmd = ["git"] + args
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def get_dir_size(path: str) -> int:
    """
    粗略统计目录大小（字节）。用于估算下载量和速度。
    """
    total = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def clone_or_update_repo(
    job: RepoJob,
    shallow: bool,
    update_existing: bool,
    timeout: Optional[int],
) -> Tuple[int, float]:
    """
    如果本地没有仓库，执行 git clone。
    如果本地有仓库：
      - update_existing=True: git fetch --all --prune --tags
      - update_existing=False: 直接视为成功
    返回 (repo_size_bytes, elapsed_seconds)。
    """
    path = job.local_path
    t0 = time.time()

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        args = ["clone"]
        if shallow:
            args += ["--depth", "1"]
        args += [job.git_url, path]
        rc, out, err = run_git(args, timeout=timeout)
        if rc != 0:
            msg = (err or out or "").strip()
            msg = msg.replace("\r", " ").replace("\n", " | ")
            if len(msg) > 300:
                msg = msg[-300:]
            raise RuntimeError(f"git clone failed: {msg}")
        # clone 完成后统计目录大小
        size = get_dir_size(path)
        elapsed = time.time() - t0
        return size, elapsed

    # 目录已存在
    git_dir = os.path.join(path, ".git")
    if not os.path.isdir(git_dir):
        raise RuntimeError(f"Target path exists but is not a git repo: {path}")

    if not update_existing:
        # 不更新，直接认为成功，大小就当前目录大小
        size = get_dir_size(path)
        elapsed = time.time() - t0
        return size, elapsed

    # 更新已有仓库
    rc, out, err = run_git(["fetch", "--all", "--prune", "--tags"], cwd=path, timeout=timeout)
    if rc != 0:
        msg = (err or out or "").strip()
        msg = msg.replace("\r", " ").replace("\n", " | ")
        if len(msg) > 300:
            msg = msg[-300:]
        raise RuntimeError(f"git fetch failed: {msg}")

    size = get_dir_size(path)
    elapsed = time.time() - t0
    return size, elapsed


# --------- Worker 逻辑 ---------

def worker_job(
    job: RepoJob,
    status_record: RepoStatus,
    shallow: bool,
    update_existing: bool,
    timeout: Optional[int],
) -> Tuple[str, bool, str, int, float]:
    """
    供线程池调用。
    返回 (key, success, error_message, repo_size_bytes, elapsed_seconds)。
    """
    key = job.key
    try:
        size, elapsed = clone_or_update_repo(
            job,
            shallow=shallow,
            update_existing=update_existing,
            timeout=timeout,
        )
        return key, True, "", size, elapsed
    except Exception as e:
        return key, False, str(e), 0, 0.0


# --------- 主流程 ---------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch clone/update GitHub repos from sampled_crates.csv with resume support."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to sampled_crates.csv generated by sampling script.",
    )
    parser.add_argument(
        "--out-dir",
        default="repos",
        help="Directory to store cloned repos (default: ./repos)",
    )
    parser.add_argument(
        "--status-file",
        default="download_status.csv",
        help="Path to status CSV for resume (default: download_status.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers (default: 8)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum attempts per repo before giving up (default: 3)",
    )
    parser.add_argument(
        "--shallow",
        action="store_true",
        help="Use shallow clone (git clone --depth 1). Default is full clone.",
    )
    parser.add_argument(
        "--no-update-existing",
        action="store_true",
        help="If set, do NOT update existing repos, just treat them as success.",
    )
    parser.add_argument(
        "--redo-success",
        action="store_true",
        help="If set, re-run even repos marked as success in status file.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Timeout for each git operation in seconds (default: 600).",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    log(f"Loading jobs from {args.csv} ...")
    jobs = load_jobs_from_sample(args.csv, out_dir)
    log(f"Total unique GitHub repos from sample: {len(jobs)}")

    log(f"Loading status from {args.status_file} ...")
    status = load_status(args.status_file, jobs)

    shallow = args.shallow
    update_existing = not args.no_update_existing
    timeout = args.timeout_seconds if args.timeout_seconds > 0 else None

    # 选择需要执行的 job
    to_run: List[Tuple[RepoJob, RepoStatus]] = []
    for key, job in jobs.items():
        st = status[key]
        if st.status == "success" and not args.redo_success:
            continue
        if st.attempts >= args.max_retries and st.status == "failed":
            continue
        to_run.append((job, st))

    log(f"Repos to process this run: {len(to_run)}")

    if not to_run:
        log("Nothing to do. All repos are either successful or exceeded max_retries.")
        return

    lock = threading.Lock()

    def update_status_and_save(key: str, success: bool, error_msg: str):
        with lock:
            st = status[key]
            st.attempts += 1
            # timezone-aware UTC 时间
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            st.last_attempt_ts = ts
            if success:
                st.status = "success"
                st.last_error = ""
            else:
                st.status = "failed"
                st.last_error = error_msg
            save_status(args.status_file, status)

    total_to_run = len(to_run)
    global_start = time.time()
    completed = 0
    succ = 0
    fail = 0
    total_bytes = 0

    log("Starting worker pool ...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_key = {}
        for job, st in to_run:
            future = executor.submit(
                worker_job,
                job,
                st,
                shallow,
                update_existing,
                timeout,
            )
            future_to_key[future] = job.key

        try:
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    key, success, error_msg, size_bytes, elapsed = future.result()
                except Exception as e:
                    success = False
                    error_msg = f"worker exception: {e}"
                    size_bytes = 0
                    elapsed = 0.0

                completed += 1
                if success:
                    succ += 1
                else:
                    fail += 1

                if size_bytes > 0 and elapsed > 0:
                    repo_mb = size_bytes / (1024 * 1024)
                    repo_speed = repo_mb / elapsed
                else:
                    repo_mb = 0.0
                    repo_speed = 0.0

                total_bytes += size_bytes
                total_elapsed = max(time.time() - global_start, 1e-6)
                total_mb = total_bytes / (1024 * 1024)
                total_speed = total_mb / total_elapsed

                # 进度行：总体进度 + 当前仓库信息 + 当前仓库平均速度 + 全局平均速度
                if success:
                    state_tag = "OK"
                else:
                    state_tag = "FAIL"

                log(
                    f"[{state_tag}] {key} | "
                    f"{completed}/{total_to_run} done "
                    f"(success={succ}, failed={fail}) | "
                    f"repo ~{repo_mb:.2f} MB @ {repo_speed:.2f} MB/s | "
                    f"total ~{total_mb:.2f} MB @ {total_speed:.2f} MB/s"
                )

                update_status_and_save(key, success, error_msg)
        except KeyboardInterrupt:
            log("Interrupted by user (Ctrl+C). Waiting for running tasks to finish...")
            raise

    pend = sum(1 for st in status.values() if st.status == "pending")
    log(f"Done. success={succ}, failed={fail}, pending={pend}")
    total_elapsed = max(time.time() - global_start, 1e-6)
    total_mb = total_bytes / (1024 * 1024)
    total_speed = total_mb / total_elapsed
    log(f"Total downloaded/processed ~{total_mb:.2f} MB in {total_elapsed:.1f} s "
        f"({total_speed:.2f} MB/s)")


if __name__ == "__main__":
    main()
