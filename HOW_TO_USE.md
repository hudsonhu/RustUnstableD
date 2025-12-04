# RustUnstableD 使用指南（简版）

## 文件速览
- `sample_crates.py`：从 crates.io 的 `db-dump.tar.gz` 做两层抽样，产出 `sampled_crates_v*.csv`。
- `analyze_crate_distribution.py`：对 dump 做下载量/反向依赖分布检查，调参时可快速看总体分布（可选）。
- `download_repos.py`：批量克隆 `sampled_crates_*.csv` 中的 GitHub 仓库，带断点续跑，状态写入 `download_status_*.csv`。
- `scan_features.py`：扫描各仓库当前 HEAD 中的 `#![feature(...)]`，写入 SQLite（`repos`、`repo_features` 表）。
- `scan_history.py`：基于 `git log -S 'feature('` 重建每个 feature 的首末出现时间，写入同一个 SQLite 的 `repo_feature_history` 表。
- `analyze_features.py`：读取 SQLite，生成分析用 CSV（`analysis_outputs/` 下的 4 个文件）。
- `build_feature_timeline.py`：从本地 `rust-lang/rust` 仓库抽取官方 feature gate 的版本/状态时间线，生成 `feature_timeline.csv`。
- `analysis_notebook.ipynb`：用 pandas/seaborn 读 `analysis_outputs/*.csv`（可选连 SQLite / `feature_timeline.csv`），绘图展示。
- 数据文件：`db-dump.tar.gz`（crates.io dump）、`sampled_crates_v3.csv`、`download_status_v3.csv`、`features_head_v3.db`、`analysis_outputs/*.csv`、`feature_timeline.csv`、`wide_table_feature.csv` 与 `wide_table_repo_feature.csv`（便捷宽表，已由上游步骤合并好）。

## 从零到 Notebook 的执行顺序
1) 环境准备  
   - 需要 Python 3.9+、git。Notebook 侧安装：`pip install pandas matplotlib seaborn jupyter`.

2) 准备 crates.io dump  
   - 放置 `db-dump.tar.gz` 于项目根目录（若无，去 https://static.crates.io/db-dump.tar.gz 下载）。

3) 采样 crate 列表  
   ```bash
   python3 sample_crates.py \
     --dump ./db-dump.tar.gz \
     --out ./sampled_crates_v3.csv \
     --target-size 1500 \
     --core-top-revdeps 300 \
     --min-downloads-noncore 100 \
     --min-latest-year-noncore 2015
   ```
   - 可选：`python3 analyze_crate_distribution.py --dump ./db-dump.tar.gz [--core-list core.csv]` 查看分布。

4) 批量克隆仓库  
   ```bash
   python3 download_repos.py \
     --csv sampled_crates_v3.csv \
     --out-dir repos \
     --status-file download_status_v3.csv \
     --workers 8 \
     --max-retries 3
   ```
   - 如已有部分仓库，本地存在即会跳过；`--redo-success` 可强制重跑。

5) 扫描 HEAD 的 nightly feature  
   ```bash
   python3 scan_features.py \
     --csv sampled_crates_v3.csv \
     --status download_status_v3.csv \
     --repos-dir repos \
     --db features_head_v3.db \
     --workers 8
   ```
   - 可用 `--max-repos 5` 先小样本试跑。

6) 扫描 Git 历史时间线  
   ```bash
   python3 scan_history.py \
     --db features_head_v3.db \
     --workers 8
   ```
   - 若只扫当前仍在用 nightly 的仓库，加 `--only-head-nightly`。

7) 汇总生成分析 CSV  
   ```bash
   python3 analyze_features.py \
     --db features_head_v3.db \
     --out-dir analysis_outputs
   ```
   - 会产出：`feature_head_summary.csv`、`feature_history_summary.csv`、`feature_lifetimes.csv`、`category_summary.csv`。

8) 生成官方 feature 时间线（可选但推荐）  
   ```bash
   python3 build_feature_timeline.py \
     --rust-repo ./rust \
     --out feature_timeline.csv
   ```
   - 需先 `git clone https://github.com/rust-lang/rust.git rust`。

9) 准备宽表（如需直接用合并后的数据）  
   - `wide_table_feature.csv`：按 feature 汇总的宽表；`wide_table_repo_feature.csv`：按 (repo, feature) 的时间线宽表，均基于上面步骤生成的原始表。

10) 打开 Notebook 做可视化  
   ```bash
   jupyter notebook analysis_notebook.ipynb
   ```
   - Notebook 默认读取 `analysis_outputs/*.csv`、`features_head_v3.db`（可选）和 `feature_timeline.csv`；确认这些文件已就绪即可运行。
