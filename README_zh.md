# RustUnstableD（中文版）

> English version: [README.md](README.md)

面向大数据视角，量化 Rust 生态对 nightly/unstable feature 的依赖：从 crates.io dump 抽样、批量克隆 GitHub 仓库、扫描 `#![feature(...)]` 在当前与历史的使用，生成时间线和汇总表，用于分析和汇报。

## 快速上手
- 环境：Python 3.9+、git；如需可视化，可装 Jupyter、pandas、seaborn。
- 放置 crates.io dump：`./db-dump.tar.gz`（https://static.crates.io/db-dump.tar.gz）。
- 典型流程（细节见 HOW_TO_USE.md）：
  1) 抽样：`python3 sample_crates.py --dump db-dump.tar.gz --out sampled_crates_v3.csv --target-size 1500 ...`
  2) 克隆：`python3 download_repos.py --csv sampled_crates_v3.csv --out-dir repos --status-file download_status_v3.csv --workers 8`
  3) 扫描 HEAD 夜ly feature：`python3 scan_features.py --csv sampled_crates_v3.csv --status download_status_v3.csv --repos-dir repos --db features_head_v3.db --workers 8`
  4) 扫描历史：`python3 scan_history.py --db features_head_v3.db --workers 8`
  5) 生成分析 CSV：`python3 analyze_features.py --db features_head_v3.db --out-dir analysis_outputs`
  6) 官方时间线（可选）：`python3 build_feature_timeline.py --rust-repo ./rust --out feature_timeline.csv`
  7) 可视化：打开 `analysis_notebook.ipynb`。

## 产出物
- SQLite：`features_head_v3.db`（`repos`、`repo_features`、`repo_feature_history`）。
- 分析 CSV：`analysis_outputs/`、`analysis_final_outputs/`（特性 head/历史计数、生命周期、分类统计、采纳事件）。
- 官方特性时间线：`feature_timeline.csv`、`feature_timeline_deep.csv`（来自 rust-lang/rust）。
- 仓库规模统计：`repo_commit_counts.csv`（提交总数、可选 Rust LOC），由 `count_commits.py` 生成。

## 关键指标（当前快照）
- 规模：1,365 个仓库（源自 1,500 抽样 crate），本地 checkout ~18 GB；总提交 717,494；Rust LOC 约 3,811 万。
- 夜ly 使用率：HEAD 有不稳定特性的仓库 296（21.7%）；历史曾使用过的 429（31.4%）。
- 特性覆盖：HEAD 上 250 个不稳定 gate；历史 545 个。
- 常用特性：`doc_cfg`（137 仓库）、`test`（84）、`doc_auto_cfg`（24）。
- 生命周期：2,343 条仓库-特性历史，寿命中位数约 345 天；已退役 1,713，仍在用 630。

## 目录速览
- `sample_crates.py`、`download_repos.py`：抽样与获取。
- `scan_features.py`、`scan_history.py`：当前/历史特性扫描。
- `analyze_features.py`：分析汇总。
- `build_feature_timeline.py`：官方特性时间线。
- `analysis_notebook.ipynb`：绘图。
- 数据产出：`sampled_crates_v3.csv`、`download_status_v3.csv`、`features_head_v3.db`、`analysis_outputs/`、`analysis_final_outputs/`、`feature_timeline*.csv`、`repo_commit_counts.csv`。

## 备注
- `db-dump.tar.gz` 体积很大，已列入 .gitignore，不要再次提交到 git；需要时本地存放或重新下载。
- 仓库历史已瘦身（.git 约 2.3 MiB）；避免将大文件入库。
- 汇报时可重点强调：抽样设计（核心 vs 非核心）、数据规模、采纳时间线、特性孵化时长等“数据故事”。

