# RustUnstableD

> 中文版请见 [README_zh.md](README_zh.md)

Quantifying how Rust crates rely on nightly/unstable features. This repo samples crates from the official crates.io dump, clones their GitHub repos, scans for `#![feature(...)]` usage at HEAD and across history, then produces timelines and summaries for analysis.

## Quick Start
- Prereqs: Python 3.9+, git; optional Jupyter for charts.
- Install deps: `pip install -r requirements.txt` (pandas/seaborn not pinned; install as needed for the notebook).
- Place the crates.io dump at `./db-dump.tar.gz` (download from https://static.crates.io/db-dump.tar.gz if missing).
- Typical end-to-end run (see details in HOW_TO_USE.md):
  1) Sample crates: `python3 sample_crates.py --dump db-dump.tar.gz --out sampled_crates_v3.csv --target-size 1500 ...`
  2) Clone repos: `python3 download_repos.py --csv sampled_crates_v3.csv --out-dir repos --status-file download_status_v3.csv --workers 8`
  3) Scan HEAD nightly features: `python3 scan_features.py --csv sampled_crates_v3.csv --status download_status_v3.csv --repos-dir repos --db features_head_v3.db --workers 8`
  4) Scan git history: `python3 scan_history.py --db features_head_v3.db --workers 8`
  5) Build analysis CSVs: `python3 analyze_features.py --db features_head_v3.db --out-dir analysis_outputs`
  6) Optional official timeline: `python3 build_feature_timeline.py --rust-repo ./rust --out feature_timeline.csv`
  7) Visualize: open `analysis_notebook.ipynb`.

## What’s Produced
- SQLite: `features_head_v3.db` with per-repo HEAD features and history (`repo_features`, `repo_feature_history`, `repos`).
- Analysis CSVs: `analysis_outputs/` and `analysis_final_outputs/` (per-feature head/history counts, lifetimes, category stats, adoption events).
- Official feature timeline: `feature_timeline.csv` / `feature_timeline_deep.csv` from the rust-lang/rust repo.
- Repo-scale stats: `repo_commit_counts.csv` (commit totals, optional Rust LOC) from `count_commits.py`.

## Key Metrics (current snapshot)
- Sample size: 1,365 repos cloned (from 1,500 sampled crates), ~18 GB checkout; total commits across repos: 717,494; Rust LOC tracked: ~38M.
- Nightly usage: 296 repos (21.7%) use unstable features on HEAD; 429 repos (31.4%) ever used unstable historically.
- Feature coverage: 250 distinct unstable gates on HEAD; 545 across history.
- Heavy hitters on HEAD: `doc_cfg` (137 repos), `test` (84), `doc_auto_cfg` (24).
- Lifetimes: 2,343 repo-feature histories; median lifetime ~345 days; 1,713 retired vs 630 still present.

## Repo Layout
- `sample_crates.py`, `download_repos.py`: sampling & acquisition.
- `scan_features.py`, `scan_history.py`: detection at HEAD and across git history.
- `analyze_features.py`: rollups into CSVs.
- `build_feature_timeline.py`: official Rust gate timeline.
- `analysis_notebook.ipynb`: plotting.
- Data artifacts: `sampled_crates_v3.csv`, `download_status_v3.csv`, `features_head_v3.db`, `analysis_outputs/`, `analysis_final_outputs/`, `feature_timeline*.csv`, `repo_commit_counts.csv`.

## Notes
- The crate dump (`db-dump.tar.gz`) is large and intentionally ignored in git; keep it locally or fetch fresh when needed.
- After history rewrite, the repository is small (~2.3 MiB .git); avoid re-adding large artifacts.
- For presentations, focus on data scale, sampling design (core vs non-core), adoption timelines, and incubation stats.

