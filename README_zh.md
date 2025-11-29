# RustUnstableD 项目说明（中文）

## 1. 项目背景与总体目标

本项目面向 NYU Big Data Management 课程的大作业，研究主题是：

> **“Rust 语言中不稳定（unstable）特性的下游使用情况与演化轨迹”**

Rust 是一门编译型系统编程语言。为了在保证语言稳定性的前提下持续演进，Rust 对很多新能力采取了“先实验、后稳定”的策略：

* 新特性首先在 **nightly**（实验通道）中以 **`#![feature(...)]` gate** 的形式提供；
* 只有在长期验证后，才会进入 **stable**（稳定版本）；
* 这意味着：**下游真实项目如果想抢先使用语言新能力，就必须显式打开这些 unstable gates**，例如：

  ```rust
  #![feature(allocator_api)]
  #![feature(generic_const_exprs)]
  ```

在生态层面，这引出了几个自然的问题：

1. **谁在用这些实验特性？**
   是系统级库？区块链？编译器工具？还是一些实验性个人项目？

2. **这些特性在项目中的“寿命”是怎样的？**
   某个 gate 被引入之后，会被长期依赖，还是短暂试用后弃用？

3. **从生态视角看，“被用得多”的特性是否会更快被稳定化？**
   虽然我们不会直接控制 Rust 官方的决策，但可以通过观察：

   * 某特性在多少仓库出现；
   * 使用时间跨度有多长；
   * 稳定前后生态的行为是否有明显变化。

本项目的总体目标是：

* 从 **crates.io 全量数据** 中抽样出一个尽量“有代表性”的 Rust 项目集合；
* 自动批量拉取这些项目的 **GitHub 仓库代码**；
* 对每个仓库进行：

  * **当前 HEAD 源码层面的 feature gate 扫描**；
  * **Git 历史层面的 feature gate 事件追踪**（首次引入、最后出现等）；
* 将这些信息统一写入 **SQLite 数据库 (`features.db`)**，再用分析脚本导出 CSV 进行可视化和统计分析。

换句话说，这是一个从数据采样 → 大规模爬取 → 静态代码分析 → Git 历史挖掘 → 统计分析的完整 pipeline。

---

## 2. 项目目录结构（当前进度）

项目根目录大致包含以下文件/目录（仅列出核心）：

```text
RustUnstableD/
  ├─ sample_crates.py         # 从 crates.io dump 中做分层抽样
  ├─ download_repos.py        # 批量克隆/更新 GitHub 仓库（带断点续跑 & 进度显示）
  ├─ scan_features.py         # 扫描各仓库 HEAD 中的 #![feature(...)] 使用
  ├─ scan_history.py          # 使用 git log -S 重建 feature gate 的历史时间线
  ├─ analyze_features.py      # 从 SQLite 中导出分析结果（CSV）
  ├─ sampled_crates.csv       # 抽样得到的 crate 列表 + GitHub 仓信息
  ├─ download_status.csv      # 仓库下载状态（可用于断点续跑）
  ├─ repos/                   # 本地克隆下来的 Git 仓库
  ├─ features.db              # 核心 SQLite 数据库（HEAD + 历史扫描结果）
  ├─ analysis_outputs/        # analyze_features.py 生成的 CSV 结果
  ├─ README.md                # 英文说明
  └─ README_zh.md             # 本文档
```

---

## 3. 环境准备

### 3.1 运行环境

* **操作系统**：macOS / Linux（理论上 Windows + WSL 也可以）
* **Python**：建议 Python 3.9+（项目中均使用标准库，无第三方依赖）
* **Git**：命令行 `git` 必须可用（用于 clone 和日志分析）
* **SQLite 命令行**（可选）：便于直接查看 `features.db` 内容

### 3.2 Python 依赖

目前所有脚本仅使用 Python 标准库模块：

* `argparse`, `csv`, `dataclasses`, `datetime`, `os`, `re`
* `subprocess`, `sqlite3`, `threading`, `time`
* `collections`, `concurrent.futures`, `statistics` 等

不需要额外 `pip install`。

---

## 4. 数据管线概览

整体流程可以分为五个阶段：

1. **从 crates.io dump 抽样 crate 列表**（`sample_crates.py`）
2. **批量克隆 GitHub 仓库**（`download_repos.py`）
3. **扫描 HEAD 源码中的 unstable feature 使用情况**（`scan_features.py`）
4. **使用 git 历史重建 feature gate 时间线**（`scan_history.py`）
5. **从 SQLite 中生成分析结果 CSV**（`analyze_features.py`）

建议按照上述顺序执行；每个阶段都可以在出现问题时单独重跑。

下面分步骤详细说明。

---

## 5. 阶段一：从 crates.io dump 抽样（`sample_crates.py`）

### 5.1 准备 crates.io 官方数据 dump

crates.io 官方会定期发布一个全量数据库 dump：

```bash
curl -L -o db-dump.tar.gz https://static.crates.io/db-dump.tar.gz
```

下载完成后，得到一个 `db-dump.tar.gz`，里面包含：

* `crates.csv`
* `crate_downloads.csv`
* `categories.csv`
* `crates_categories.csv`
* `versions.csv`
* `default_versions.csv`
* 以及其他若干辅助表

**注意**：如果直接用 Python 的 `csv` 读时遇到：

```text
field larger than field limit (131072)
```

我们在 `sample_crates.py` 中已经处理好了，会放大 `csv.field_size_limit`，确保可以读取长字段。

### 5.2 抽样策略（简要）

在 `sample_crates.py` 中，我们采用了下面的过滤和抽样逻辑（为了既控制工作量，又保持一定代表性）：

1. **时间过滤**：仅保留 **2020 年之后** 有版本发布的 crates

   * 通过 `versions.created_at` 判断
   * 对应 CLI 参数：`--min-year 2020`

2. **下载量过滤**：仅保留下载总量 **≥ min_downloads** 的 crates

   * 保证我们研究的目标是“有一定使用量”的 crate
   * 比如：`--min-downloads 100`

3. **类别信息**：通过 `categories.csv` + `crates_categories.csv`

   * 为每个 crate 提取一个或多个 *top-level category*（如 `command-line-utilities`、`cryptography` 等）
   * 若无类别，则归为 `uncategorized`

4. **分层抽样**（stratified sampling）：

   * 在每个 top-level category 内，再按下载量 quantile 分层（简单分成 low / mid / high 三档）；
   * 在各个 strata 内随机抽取一定数量，整体目标规模为 `target-size`（比如 700 个）；
   * 这样可以避免全部样本被“极热门 crates”占满，同时保留中长尾项目。

总体思路：**样本既突出生态里有一定影响力的 crates，又不完全被 top 10 热门项目主导。**

### 5.3 运行命令示例

在项目根目录运行：

```bash
python3 sample_crates.py \
  --dump ./db-dump.tar.gz \
  --out ./sampled_crates.csv \
  --target-size 700 \
  --min-year 2020 \
  --min-downloads 100
```

运行结束后，会打印类似：

```text
[sample_crates] Candidate crates after filters: 144943
[sample_crates] Download quantiles: q50=5030, q80=24578
[sample_crates] Strata count: 173
[sample_crates] Oversampled 999 > 700, trimming down.
[sample_crates] Sample CSV written to ./sampled_crates.csv

===== SAMPLING SUMMARY =====
Total candidate crates after filters: 144943
Sample size: 700

Top-level categories in sample (top 15):
  uncategorized              266
  command-line-utilities      34
  api-bindings                24
  ...
Popularity strata in sample:
  high  : 178
  low   : 304
  mid   : 218
============================
```

`sampled_crates.csv` 是后续所有步骤的输入基础，里面主要包含：

* crate 名称；
* crates.io 上记录的 GitHub 仓库地址（解析出 `owner/repo`）；
* 下载量/类别等元数据。

---

## 6. 阶段二：批量克隆 GitHub 仓库（`download_repos.py`）

### 6.1 设计目标

* 从 `sampled_crates.csv` 中提取 **唯一的 GitHub 仓库列表**（按 `owner/repo` 去重）；
* 使用多线程并行执行 `git clone` / `git fetch`；
* 通过一个 **`download_status.csv`** 记录每个仓库的下载状态，用于：

  * 断点恢复；
  * 重跑时避免重复 clone；
  * 分析失败原因（仓库删掉、私有化等）。

### 6.2 状态文件字段（`download_status.csv`）

每一行代表一个 GitHub 仓库，典型字段包含：

* `repo_full_name`：`owner/repo`
* `owner`：GitHub 用户/组织名
* `name`：仓库名
* `crate_name`：来源 crate 名（有时 1 个仓对应多个 crate，存其中一个）
* `status`：

  * `success`：下载成功
  * `failed`：下载尝试过但失败
* `attempt_count`：尝试次数
* `last_error`：最近一次失败的错误信息（如 404、权限问题）
* `last_attempt_ts`：最近一次尝试时间（UTC ISO 格式）

### 6.3 运行命令示例

在项目根目录：

```bash
python3 download_repos.py \
  --csv sampled_crates.csv \
  --out-dir repos \
  --status-file download_status.csv \
  --workers 8 \
  --max-retries 3
```

参数说明：

* `--csv`：`sample_crates.py` 产生的 crate 样本；
* `--out-dir`：本地 clone 下来的仓库存放目录（例如 `repos/owner/repo`）；
* `--status-file`：下载状态 CSV 文件；
* `--workers`：并行 worker 数（建议 ≤ CPU 逻辑核心数）；
* `--max-retries`：失败后最多重试次数。

运行时会在终端输出进度，例如：

```text
[20:06:32] Loading jobs from sampled_crates.csv ...
[20:06:32] Total unique GitHub repos from sample: 679
[20:06:32] Loading status from download_status.csv ...
[20:06:32] Repos to process this run: 45
[20:06:32] Starting worker pool ...
[20:06:32] [FAIL] CodeWolf33/rust_bmp | 1/45 done (success=0, failed=1) | repo ~0.00 MB @ 0.00 MB/s | total ~0.00 MB @ 0.00 MB/s
...
[20:16:36] Done. success=0, failed=45, pending=0
[20:16:36] Total downloaded/processed ~0.00 MB in 604.4 s (0.00 MB/s)
```

后续如果手动 `git clone` 了一些失败仓库，也可以直接往 `repos/owner/repo` 里放，只要路径存在，扫描阶段会自动识别。

---

## 7. 阶段三：扫描 HEAD 源码中的 unstable features（`scan_features.py`）

### 7.1 目标

这一步是在 **当前 HEAD 源码** 上，统计：

* 每个仓库是否启用了任何 unstable feature gate；
* 每个 `feature_name` 在多少仓库中出现；
* 每个 `(repo, feature)`：

  * 有多少个 crate 级的 `#![feature(...)]` / `#![cfg_attr(..., feature(...))]` 属性；
  * 出现在多少个文件中，示例路径若干。

结果写入 SQLite 数据库 `features.db` 中：

* `repos` 表：仓库级元信息；
* `repo_features` 表：HEAD 上的 feature gate 使用情况。

### 7.2 数据库表结构（核心）

**`repos` 表**（简化示意）：

* `key`：`owner/repo`，主键；
* `owner`：GitHub owner；
* `name`：仓库名；
* `crate_name`：关联的主 crate 名；
* `path`：本地存放路径（如 `repos/owner/repo`）；
* `top_categories`：逗号分隔类别列表（来自 crates.io）；
* `downloads`：下载量；
* `head_scanned`：是否已经完成 HEAD 扫描；
* `last_head_scan_ts`：最近一次 HEAD 扫描时间。

**`repo_features` 表**（HEAD 视角）：

* `key`：`owner/repo`；
* `feature_name`：例如 `allocator_api`、`generic_const_exprs` 等；
* `attr_count`：该 feature gate 在当前 HEAD 出现的 attribute 次数；
* `file_count`：涉及的文件数；
* `example_paths`：若干示例文件路径（逗号分隔或 JSON 串）。

### 7.3 扫描逻辑简述

脚本会：

1. 从 `sampled_crates.csv` 和 `download_status.csv` 读取适用的仓库列表：

   * 默认只扫描 `status = success` 的仓库；
   * 可通过参数指定 `--include-failed`，在本地路径存在时也扫描。

2. 在每个仓库里，只扫描 `.rs` 源码文件，寻找 crate-level inner attributes：

   * 形如 `#![feature(...)]`
   * 或 `#![cfg_attr(..., feature(...))]`

3. 使用正则从 attribute 里解析出 feature 名字，并累积计数。

### 7.4 运行命令示例

例如只测 5 个仓库：

```bash
python3 scan_features.py \
  --csv sampled_crates.csv \
  --status download_status.csv \
  --repos-dir repos \
  --db features.db \
  --workers 4 \
  --max-repos 5
```

之后可以逐步放大到全部仓库：

```bash
python3 scan_features.py \
  --csv sampled_crates.csv \
  --status download_status.csv \
  --repos-dir repos \
  --db features.db \
  --workers 8
```

成功后，可以用 `sqlite3 features.db` 快速查看：

```sql
-- 有多少仓库在 HEAD 上启用了任何 unstable feature？
SELECT COUNT(DISTINCT key) FROM repo_features;

-- 哪些 feature 在 HEAD 上出现得最多？
SELECT feature_name, COUNT(*) AS repo_count
FROM repo_features
GROUP BY feature_name
ORDER BY repo_count DESC
LIMIT 40;
```

---

## 8. 阶段四：基于 Git 历史重建 feature gate 时间线（`scan_history.py`）

### 8.1 动机

仅看 HEAD 只能回答：

> “**现在** 哪些仓库还在用哪些 unstable feature？”

但我们更关心的是“**历史趋势**”：

* 某个 feature gate 是从什么时候开始在生态里被使用的？
* 在单个仓库内，它的使用寿命有多长？
* 有些 feature 在稳定之后，项目是否迅速删掉 gate、迁移到 stable？

因此需要利用 **Git 历史** 来构建时间线。

### 8.2 核心思路：基于 `git log -S 'feature('` 的事件流

对每个仓库，我们执行：

```bash
git log --reverse -p -S 'feature(' --format=__COMMIT__%H %ct -- '*.rs'
```

含义：

* `-S 'feature('`：所谓 pickaxe，选出“`feature(` 字符串计数发生变化”的 commit；
* `-p`：输出 diff，方便在 `+` / `-` 行里找 `#![feature(...)]`；
* `--reverse`：从最早的 commit 往后看，便于记录 first_seen；
* `--format=__COMMIT__%H %ct`：每个 commit 前加一个标记，带上 hash 和 Unix 时间戳。

脚本对每个 commit：

* 收集所有新增行 (`+...`) 中新增的 feature 名；
* 收集所有删除行 (`-...`) 中删除的 feature 名；
* 用一个计数器 `feature_counts[feature]` 表示当前源码中该 feature gate 是否“存在”；
* 当计数从 0→1 时，记录 `first_seen_commit/date`；
* 当计数从 >0→0 时，记录 `last_seen_commit/date`（gate 被彻底移除的最后一次 commit）。

最后再结合 HEAD 扫描结果判断：

* 如果 HEAD 仍然有某 feature gate，则 `still_present = 1`，`last_seen` 取当前 HEAD；
* 否则为 `still_present = 0`，`last_seen` 为最后一次移除 gate 的 commit。

### 8.3 历史表结构 `repo_feature_history`

在 `features.db` 中新增表：

```sql
CREATE TABLE IF NOT EXISTS repo_feature_history (
  key              TEXT,
  feature_name     TEXT,
  first_seen_commit TEXT,
  first_seen_date   TEXT,
  last_seen_commit  TEXT,
  last_seen_date    TEXT,
  still_present     INTEGER,
  PRIMARY KEY (key, feature_name),
  FOREIGN KEY (key) REFERENCES repos(key)
);
```

每行表示：某个仓库中的某个 feature gate 的历史使用概况。

### 8.4 运行命令示例

1. **只扫描当前 HEAD 上已经有 nightly 的仓库：**

```bash
python3 scan_history.py \
  --db features.db \
  --only-head-nightly \
  --workers 4 \
  --max-repos 50      # 先小规模测试
```

2. **对所有仓库进行历史扫描：**

```bash
python3 scan_history.py \
  --db features.db \
  --workers 8
```

脚本会逐仓库写入 `repo_feature_history` 表。如果某仓库重跑，会先删除该 `key` 的旧记录，再写入新的结果，方便迭代。

完成后，可以用一些简单 SQL 查看：

```sql
-- 有多少仓库历史上曾经打开过任何一个 feature gate？
SELECT COUNT(DISTINCT key) FROM repo_feature_history;

-- 按 feature 看历史上使用过它的仓库数
SELECT feature_name, COUNT(*) AS ever_repo_count
FROM repo_feature_history
GROUP BY feature_name
ORDER BY ever_repo_count DESC
LIMIT 40;

-- 查看某个 heavy-nightly 仓库的所有 feature 时间线
SELECT *
FROM repo_feature_history
WHERE key = 'rust-lang/rustfmt'
ORDER BY feature_name;
```

---

## 9. 阶段五：从数据库导出分析结果（`analyze_features.py`）

### 9.1 目标

`analyze_features.py` 用来对 `features.db` 进行 **二次加工**，输出适合用 Excel / Numbers / pandas 处理的 CSV 文件。主要做四类汇总：

1. 整体生态概况（HEAD vs EVER）
2. 每个 feature 的使用热度统计（HEAD / 历史）
3. 每个 feature 的“寿命”统计（在各仓库中的使用时长分布）
4. 不同类别（top_categories）中 nightly 使用比例

### 9.2 输出目录与文件

脚本会在项目根目录下创建：

```text
analysis_outputs/
  ├─ feature_head_summary.csv
  ├─ feature_history_summary.csv
  ├─ feature_lifetimes.csv
  └─ category_summary.csv
```

含义：

* `feature_head_summary.csv`

  * 每行：`feature_name`, `head_repo_count`
  * 描述：HEAD 上使用该 gate 的仓库数排行榜。

* `feature_history_summary.csv`

  * 每行：`feature_name`, `ever_repo_count`
  * 描述：历史上曾经使用过该 gate 的仓库数排行榜。

* `feature_lifetimes.csv`
  对每个 `feature_name` 计算：

  * `num_repos`：历史上使用过它的仓库数；
  * `num_still_present`：这些仓库里目前仍然使用它的数量；
  * `num_retired`：历史上用过但现在不用了的仓库数；
  * `min_first_seen_date` / `max_first_seen_date`：第一次被引入的时间范围；
  * `min_last_seen_date` / `max_last_seen_date`：被最后使用的时间范围；
  * `avg_lifetime_days` / `median_lifetime_days`：在有完整首末时间的仓库中，该特性 gate 的平均/中位使用寿命（天）。

* `category_summary.csv`
  按 `repos.top_categories` 聚合：

  * `category`：顶层类别；
  * `total_repos`：该类别中的仓库数量；
  * `head_nightly_repos`：HEAD 上使用任何 nightly feature 的仓库数；
  * `ever_nightly_repos`：历史上曾经使用过 nightly feature 的仓库数；
  * `head_nightly_ratio` / `ever_nightly_ratio`：对应比例（便于判断哪些领域更偏向 nightly）。

### 9.3 运行命令示例

在项目根目录：

```bash
python3 analyze_features.py
```

终端会先输出整体概况，例如：

```text
[22:30:01] === BASIC OVERVIEW ===
[22:30:01] Total repos in sample: 679
[22:30:01] Repos with HEAD-level unstable features (repo_features): 103
[22:30:01] Repos that ever used unstable features (repo_feature_history): 210
[22:30:01] HEAD-nightly ratio: 103/679 ≈ 15.17%
[22:30:01] Ever-nightly ratio: 210/679 ≈ 30.92%
...
```

然后在 `analysis_outputs/` 下生成 CSV 文件。

接下来就可以用 Excel / Numbers / Python notebook 做进一步分析和画图。

---

## 10. 方法局限与注意点（写报告时可用）

1. **只观察 feature gate，而非语言特性本身的所有使用**

   * 我们的度量是基于显式的 `#![feature(...)]` 属性；
   * 对于已经稳定且 gate 被彻底清理的特性（例如 `async_fn_in_trait`），
     在 HEAD 和很多“晚期 commit”里都不会再出现 gate，因此我们的数据**低估**了其真实使用范围。

2. **样本偏向 2020 年以后、下载量 ≥ 100 的 crates**

   * 这让我们更关注“当代生态中的主流项目”，但会漏掉很多早期老牌项目；
   * 某些重度实验项目（例如很多 2018–2019 就开始的 async 框架）可能不在样本中。

3. **Git 历史扫描基于 `-S 'feature('`，只捕获“feature 字符串计数变化的 commit”**

   * 若某个 crate 在早期 commit 中就存在大量 feature gate，然后一直小修小改，
     没有增删 `feature(` 字符串，则中间那段历史不会被 pickaxe 捕获；
   * 我们记录的是“first_seen”和“最后一次可观测到 gate 的变化”的时间，并假设在这段时间内 gate 一直存在/可用，这是一种近似。

4. **解析逻辑是基于正则的静态分析，可能漏掉极端格式**

   * 我们只匹配 crate-level inner attributes (`#![...]`) 中的 `feature(...)`；
   * 如果有极为奇怪的宏展开形式、或者自定义 attribute 里嵌入了字符串 `"feature("`，可能会被误判或忽略；
   * 这些情况在真实项目中相对少见，对整体统计影响有限。

5. **仓库下载可能失败**

   * 部分仓库可能被删除、设为私有、或网络问题导致无法 clone；
   * 这些仓库在后续分析中被自然排除，有一定样本流失，但比例较小。

在报告中，这些都可以写进 “Threats to Validity / Limitations” 一节，显示我们对方法边界有清晰认识。

---

## 11. 后续 TODO / 可能扩展方向

1. **更精细的 per-repo 时间线可视化**

   * 对选定的几家 heavy-nightly 仓库（如 `rust-lang/rustfmt`、某些区块链 SDK）画出
     “特性随时间引入/移除”的折线图或甘特图。

2. **与 Rust 官方稳定时间线对齐**

   * 手动整理我们关心的 8–12 个核心特性（如 `allocator_api`, `generic_const_exprs`, `type_alias_impl_trait`, `specialization` 等）的 RFC/稳定版本时间；
   * 结合 `feature_lifetimes.csv`，分析生态在特性稳定前后是否有明显 adoption 峰值或退潮。

3. **增加简单的仓库级“依赖强度”指标**

   * 比如综合 `attr_count`、`file_count`、`feature_lifetimes` 等，给每个 `(repo, feature)` 算一个粗略的“feature 依赖强度分数”；
   * 用于区分“只是开了 gate，几乎没用到”的情况和“强耦合”的情况。

4. **更严格去噪 feature 名称**

   * 目前历史扫描会把任何 `feature(...)` 中出现的标识符都当成 feature 名，比如会看到 `Serde`, `a` 等噪声；
   * 后续可以引入一份官方 feature gate 列表（从 rust-lang/rust 仓库导出），只保留其中的合法名字。
