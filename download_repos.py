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
from datetime import datetime
import threading


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

    # 特例：SSH 形式 git@github.com:owner/repo.git
    if s.startswith("git@github.com:"):
        tail = s[len("git@github.com:"):]
        # 去掉可能的 .git 和尾部杂项
        tail = tail.split("#", 1)[0].split("?", 1)[0]
        if tail.endswith(".git"):
            tail = tail[:-4]
        parts = tail.strip("/").split("/")
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1]
        normalized = f"https://github.com/{owner}/{repo}.git"
        return owner, repo, normalized

    # 正常 URL / 裸域名
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

    # 去掉 .git
    if repo.endswith(".git"):
        repo = repo[:-4]

    normalized = f"https://github.com/{owner}/{repo}.git"
    return owner, repo, normalized


# --------- 读取 sampled_crates.csv 并构建 RepoJob ---------

def load_jobs_from_sample(csv_path: str, out_dir: str) -> Dict[str, RepoJob]:
    """
    从 sampled_crates.csv 读取数据，根据 repository 字段解析 GitHub 仓库，
    聚合成 RepoJob（一个 repo 可能对应多个 crate）。
    返回 dict: key -> RepoJob
    """
    jobs: Dict[str, RepoJob] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = ["name", "repository", "top_category", "pop_stratum"]
        for c in required_cols:
            if c not in reader.fieldnames:
                raise RuntimeError(f"sample CSV missing column: {c}")

        for row in reader:
            crate_name = (row.get("name") or "").strip()
            repo_url = (row.get("repository") or "").strip()
            top_category = (row.get("top_category") or "").strip() or "unknown"
            pop_stratum = (row.get("pop_stratum") or "").strip() or "unknown"

            parsed = parse_github_repo(repo_url)
            if parsed is None:
                # 非 github 或格式怪，先跳过
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
    把 status 写回 CSV。每次写全量，反正几百行很小。
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
            writer.writerow(asdict(status[key]))
    os.replace(tmp_path, status_path)


# --------- Git 操作 ---------

def run_git(args: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
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
    )
    return result.returncode, result.stdout, result.stderr


def clone_or_update_repo(
    job: RepoJob,
    shallow: bool,
    update_existing: bool,
) -> None:
    """
    如果本地没有仓库，执行 git clone。
    如果本地有仓库：
      - update_existing=True: git fetch --all --prune --tags
      - update_existing=False: 直接视为成功
    出错则抛异常，让上层记录失败。
    """
    path = job.local_path
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        args = ["clone"]
        if shallow:
            args += ["--depth", "1"]
        args += [job.git_url, path]
        rc, out, err = run_git(args)
        if rc != 0:
            raise RuntimeError(f"git clone failed: {err.strip()[-500:]}")
        return

    # 目录已存在
    git_dir = os.path.join(path, ".git")
    if not os.path.isdir(git_dir):
        raise RuntimeError(
            f"Target path exists but is not a git repo: {path}"
        )

    if not update_existing:
        # 用户指定不更新，直接视为 success
        return

    # 更新已有仓库
    rc, out, err = run_git(["fetch", "--all", "--prune", "--tags"], cwd=path)
    if rc != 0:
        raise RuntimeError(f"git fetch failed: {err.strip()[-500:]}")
    # 可以慎重点，只 fetch，不强制 pull；以后分析用 fetch 足够
    return


# --------- Worker 逻辑 ---------

def worker_job(
    job: RepoJob,
    status_record: RepoStatus,
    shallow: bool,
    update_existing: bool,
) -> Tuple[str, bool, str]:
    """
    供线程池调用。
    返回 (key, success, error_message)。
    """
    key = job.key
    try:
        clone_or_update_repo(job, shallow=shallow, update_existing=update_existing)
        return key, True, ""
    except Exception as e:
        return key, False, str(e)


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

    # 选择需要执行的 job
    to_run: List[Tuple[RepoJob, RepoStatus]] = []
    for key, job in jobs.items():
        st = status[key]
        # 判断是否需要执行
        if st.status == "success" and not args.redo_success:
            continue
        if st.attempts >= args.max_retries and st.status == "failed":
            # 超过最大重试次数，跳过
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
            st.last_attempt_ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            if success:
                st.status = "success"
                st.last_error = ""
            else:
                st.status = "failed"
                st.last_error = error_msg
            save_status(args.status_file, status)

    # 线程池执行
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_key = {}
        for job, st in to_run:
            future = executor.submit(
                worker_job,
                job,
                st,
                shallow,
                update_existing,
            )
            future_to_key[future] = job.key

        try:
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    key, success, error_msg = future.result()
                except Exception as e:
                    # 理论上不会到这里，因为 worker 已 catch，保险起见
                    success = False
                    error_msg = f"worker exception: {e}"
                if success:
                    log(f"[OK] {key}")
                else:
                    log(f"[FAIL] {key} | {error_msg}")
                update_status_and_save(key, success, error_msg)
        except KeyboardInterrupt:
            log("Interrupted by user (Ctrl+C). Waiting for running tasks to finish...")
            # 线程池会在 with 退出时等待已提交任务
            # 状态文件已经在每个任务完成时持续刷盘
            raise

    # 总结一下
    succ = sum(1 for st in status.values() if st.status == "success")
    fail = sum(1 for st in status.values() if st.status == "failed")
    pend = sum(1 for st in status.values() if st.status == "pending")
    log(f"Done. success={succ}, failed={fail}, pending={pend}")


if __name__ == "__main__":
    main()
