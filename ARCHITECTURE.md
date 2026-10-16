# Architecture

A map of this repository: what the directories hold, how the modules depend on each other, the
conventions the code relies on, and where a change is likely to hurt.

## Structure

The repository is a four-step batch pipeline plus two side branches. Every step is a script or a
notebook that walks a directory of monthly CSVs, writes another directory of monthly CSVs, and
exits; there is no orchestrator, no configuration file and no CLI. Steps are selected by editing
`bool` flags at the top of a block (`K_mean_Save = True`, `dbscan_Save = False`, ...).

```
repo root
├── 0_preprocessing.ipynb        winsorise the next-month return matrix
├── 1_perform_clustering.py      characteristics -> monthly cluster labels
├── 2_invest.py                  cluster labels -> monthly long/short tables
├── 3_check_performance.ipynb    long/short tables -> return series + statistics
├── 4_analyze_portfolio.ipynb    return series -> per-sub-period statistics
├── cl_outlier_filter.ipynb      side branch: re-label low-confidence firms as outliers
├── ppo_threshold.ipynb          side branch: RL agent picks the spread threshold
├── src/                         the library the above import
└── data/, results/              inputs, intermediates and outputs (all CSV)
```

Numbered entry points are the main pipeline and run in order. Unnumbered notebooks are side
branches that hang off it.

### `src/` (7 files, 688 NLOC by `lizard`'s count; the two root scripts add 297 more)

| Path | Role |
| --- | --- |
| `src/features.py` | The shared base. Reading a monthly CSV, finding the momentum column prefix, PCA on the characteristics, merging monthly long/short tables into one wide matrix, turning that into a return series, and plotting/saving it. Imported with `import *` by everything else. |
| `src/clustering.py` | `Clustering` — k-means, DBSCAN and agglomerative clustering of one monthly cross section, each with its own outlier rule. Produces a list of lists of firm names. |
| `src/invest_table.py` | `Invest_Table` — turns clusters into a per-firm long/short table (`ls_table`), plus the decile-reversal benchmark (`reversal_table`). |
| `src/cointegration.py` | `cointegration` — the textbook pairs-trading alternative: Engle-Granger, ADF and KPSS filters over all firm pairs in a month, parallelised with `multiprocessing.Pool(5)`. |
| `src/metrics.py` | `metrics` — the performance table (annualised mean/std, Sharpe, Sortino, profit factor, max drawdown, Calmar, Newey-West t-statistics, monthly distribution). |
| `src/ppo_env.py` | `SimpleEnv`, a `gym.Env` wrapper around one month of the strategy. Used only by `ppo_threshold.ipynb`. |
| `src/__init__.py` | Empty; makes `src` a package. |

### `data/` — 42,356 files, 5.3 GiB uncompressed

Almost the whole repository by count and by size is monthly CSV shards: **one CSV per month per
configuration**. The US panel runs 1990-01 to 2022-08 with 1,845 to 3,609 firms per month, so 392 files per configuration, and there
are 106 such configurations of cluster results (106 x 392 = 41,552 files), every one of them
named `YYYY-MM.csv` with no stray files.

| Path | Files | Size | Content |
| --- | --- | --- | --- |
| `data/characteristics_US/` | 392 | 1.8 GiB | `YYYY-MM.csv`, 1990-01 to 2022-08. One row per firm (1,846–1,954 rows), 128 columns: `firms` + 78 firm characteristics + `mom1`…`mom49`. |
| `data/characteristics_UK/` | 402 | 419 MiB | Same shape for the UK, 1990-02 to 2023-07, 60 columns: 10 characteristics + `Mom_1`…`Mom_49`. |
| `data/Clustering_Result/K_mean/{10,20,30,40,50,100}/` | 2,352 | 166 MiB | Output of `1_perform_clustering.py`: `firms, mom1, clusters` per month. One sub-directory per K. |
| `data/Clustering_Result/DBSCAN/{0.1…0.9}/` | 3,528 | 247 MiB | Same, one sub-directory per eps percentile. |
| `data/Clustering_Result/Agglomerative/{0.1…0.9}/` | 3,528 | 255 MiB | Same, one sub-directory per distance percentile. |
| `data/Clustering_Result/Contrastive_Learning/` | 3,136 | 224 MiB | Cluster labels from a contrastive-learning encoder, for 60/70/80/90 clusters, plus a `CL_Pre` filtered variant of the same four: 8 configurations. **The code that produced these is not in this repository.** |
| `data/Clustering_Result/Contrastive_Learning2/` | 27,440 | 1.9 GiB | The same for 10…70 clusters, crossed with nine confidence thresholds `CL_pre_0.1`…`CL_pre_0.9` written by `cl_outlier_filter.ipynb`: 7 + 9 × 7 = 70 configurations, 65% of all files in the repository. |
| `data/Clustering_Result/Best/` | 1,568 | 111 MiB | One chosen configuration per clustering family, copied here so `2_invest.py` and `ppo_threshold.ipynb` can read them side by side. Byte-identical to `Agglomerative/0.6`, `DBSCAN/0.1`, `K_mean/50` and `Contrastive_Learning2/CL_pre_0.8/40_characteristics_us` (verified by comparing files). |
| `data/Momentum1_{Original,Winsorized}.csv` | 2 | 61 MiB | The realised-return matrix, 9,749 firms × 392 months. `Winsorized` is what every backtest reads: `0_preprocessing.ipynb` sets anything above 1.0 to 1.5 and anything below -0.95 to -0.95. The `scipy` `winsorize` call in the same notebook is applied to a frame that is never saved. |
| `data/ETC/` | 8 | 117 MiB | Benchmarks and raw return matrices: `sp500_return_div.csv` (the benchmark used in the headline table), `sp500_*.csv`, `FTSE_return.csv`, `momentum_us.csv`, `_momentum.csv`. |

`data/LS_Result/` is in `.gitignore` and is **not** committed. It is the per-month long/short
tables that `2_invest.py` writes and `3_check_performance.ipynb` reads, i.e. the one intermediate
stage that has to be regenerated before the performance notebook can run.

### `results/` — 69 files, 2.6 MiB

| Path | Content |
| --- | --- |
| `results/Total_Result/Result/*_result_modified.csv` | Monthly return series, one row per configuration, 391 columns (1990-02 … 2022-08). |
| `results/Total_Result/Result/*_statistcs_modified.csv` | The performance table for the same configurations (filename typo is in the committed data and in `src/features.py`). |
| `results/Total_Result/{L,S}_Result/` | The same for the long-only and short-only legs (`save = 'L'` / `'S'` in `3_check_performance.ipynb`). |
| `results/Total_Result/period_analysis.csv` | Output of `4_analyze_portfolio.ipynb`: seven metrics × seven sub-periods × six strategies. |
| `results/Individual_Result/<family>/<config>.csv` | Per-month diagnostics of a clustering run: firms traded, outliers, size of the two largest clusters, number of clusters. For `K_mean/50`: 2,755 firms per month on average, 59% of them in the outlier bucket, 285 traded. |
| `results/images/` | Eight PNGs from `ppo_threshold.ipynb` (cumulative return before/after RL, and the histogram of chosen thresholds). |
| `results/PPO.csv` | A single 391-month return series from the RL branch. |

## Dependencies

External: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `statsmodels` for the pipeline;
`torch` and `gym` for `ppo_threshold.ipynb` only. See `requirements.txt` for the version bounds the
code forces.

Internal fan-in — `src/features.py` is the hub; every other module and every entry point imports it:

```
             0_preprocessing      1_perform_clustering      2_invest       3_check_performance   4_analyze_portfolio
                    │                      │                   │                   │                    │
                    │                      ▼                   ▼                   │                    │
                    │             src.clustering    src.cointegration               │                    │
                    │                      │         src.invest_table               │                    │
                    │                      │                   │                   │                    │
                    └──────────────────────┴─────────┬─────────┴───────────────────┴────────────────────┘
                                                     ▼
                                             src.features  ◄──────────►  src.metrics
                                                     ▲                    (mutual import)
                                                     │
                                       src.ppo_env ──┘  ◄── ppo_threshold
```

| Module | Imported by |
| --- | --- |
| `src.features` | 5 modules + 6 entry points |
| `src.metrics` | `src.features`, `4_analyze_portfolio` |
| `src.invest_table` | `2_invest` |
| `src.clustering` | `1_perform_clustering` |
| `src.cointegration` | `2_invest` |
| `src.ppo_env` | `ppo_threshold` |

**`src.features` and `src.metrics` import each other.** `features` does `from src import metrics as M`
at line 6 and `metrics` answers with `from src.features import *`. This resolves today only because
of ordering luck: whichever module is imported first is still half-initialised when the other
`import *`s it, and both only need the other's names at call time (`M.metrics` inside
`save_and_plot_result`, `np`/`pd` inside `metrics`). Both import orders were checked and both work.
It is still the most fragile thing in the repository: adding a module-level use of `M` in
`features.py`, or of any `features` function in `metrics.py`, will break it with an
`AttributeError`/`ImportError` that points nowhere near the cause.

Six of the seven `src` modules start with `from src.features import *`, so `os`, `warnings`, `np`,
`pd`, `plt` and every helper arrive implicitly. There is no module that declares its own imports.

## Conventions

- **Monthly CSV shards.** The unit of data is one month of one configuration, named `YYYY-MM.csv`.
  Every step is a loop over `sorted(os.listdir(input_dir))`, which is why the filenames must be
  zero-padded, and why the pipeline is restartable only at month granularity.
- **Run from the repository root.** All paths are relative (`data/...`, `results/...`), so the
  working directory has to be the repository root. One exception: `cl_outlier_filter.ipynb` uses
  `../data/...` and expects to be run from a sub-directory. That inconsistency predates this
  cleanup and was left as it was rather than silently changed.
- **Output directories must already exist.** `to_csv(os.path.join(output_dir, file))` never creates
  its parent, so `data/LS_Result/<family>/<config>/` has to be made by hand before a run.
- **`bool` flags instead of arguments.** `1_perform_clustering.py` and `2_invest.py` are flat
  scripts whose blocks are enabled by editing `K_mean_Save`, `dbscan_Save`, `Best`, `cointegration`,
  ... As committed, only the k-means block of each is `True`, and `1_perform_clustering.py` has
  `test_Set = [40]`, so running it reproduces one configuration, not the grid.
- **Cluster index 0 means "outlier".** Every clustering method puts its rejects in the first
  sub-list, and `ls_table` forces `Long Short = 0` for cluster 0. A method that happens to find no
  outliers inserts an empty list to keep the convention (`src/clustering.py`, `perform_HA`).
- **Momentum column prefix is sniffed, not assumed.** `momentum_prefix_finder` exists because the
  US panel names the column `mom1` and the UK panel names it `Mom_1`.
- **Returns are log returns throughout.** Monthly position returns are capped at -0.30 before
  `log(1+r)`; a firm that is held but has no return next month is charged -0.5 as a delisting proxy
  (`src/features.py`, `product_LS_Table`).
- **The performance metrics are not the textbook ones.** `src/metrics.py` annualises as
  `exp(12·mean)-1` and `exp(√12·std)-1`, and reports Sharpe as the ratio of those two with **no
  risk-free rate**. Max drawdown is computed on `exp(cumsum(log returns))`. The numbers are
  internally consistent but are not comparable to Sharpe ratios quoted elsewhere.
- **391 months is hard-coded in one place.** `metrics.cal_monthly_statistics` regresses on
  `X = np.ones(391)` rather than on `len(row)`, so the Newey-West t-statistic it reports is only
  valid for a 391-month series and raises on any other length. The UK period lists in
  `4_analyze_portfolio.ipynb` go up to 402 months.
- **The `t-statistic` row is relative to the second-to-last column.** `cal_t_statistics` tests every
  column against `result_modified.iloc[:, -2]`, whatever that happens to be. In
  `Best_statistcs_modified.csv` that is `Reversal`; in `Best2_statistcs_modified.csv`, which holds
  the same return series, it is `K-mean`. The two files therefore disagree on t-statistics while
  agreeing on returns. Read that row knowing what it is comparing against.
- **Naming.** Python files and notebooks are `lower_snake_case`; classes keep the names the call
  sites use (`Clustering`, `Invest_Table`, `cointegration`, `metrics` — the last two are lower-case
  classes, left alone to avoid touching every call site). Committed result filenames keep their
  original spelling, typos included, because notebooks read them by name.

## Impact

Where a change propagates, roughly in order of blast radius.

- **`src/features.py` touches everything.** It is imported with `import *` by all six other modules
  and all seven entry points, so it also supplies their `np`/`pd`/`os`/`plt`. Renaming anything in
  it, or adding a name that shadows one of those, changes every file in the repository.
- **The `features` ↔ `metrics` cycle** described above. Any new module-level cross-reference breaks
  it. Untangling it (moving the shared `import`s into a third module that neither imports) is the
  single highest-value refactor left, and it cannot be verified without a run.
- **Directory layout is the interface.** There is no path configuration: `data/Clustering_Result/
  <family>/<config>/YYYY-MM.csv` and `data/LS_Result/<family>/<config>/YYYY-MM.csv` are hard-coded
  in both scripts and three notebooks. Moving or renaming a data directory means editing every
  entry point, which is what this cleanup had to do when `Database/` became `data/` and `File/`
  became `results/`.
- **Complexity ranking.** `lizard` over the 9 Python files (41 functions, 985 NLOC; no function
  exceeds the default thresholds). Ranked by `CRAP_max = cc² + cc`, the ceiling a function's CRAP
  score would reach with zero test coverage:

  | CRAP_max | CC | NLOC | Function |
  | --- | --- | --- | --- |
  | 182 | 13 | 63 | `product_LS_Table` — `src/features.py` |
  | 182 | 13 | 54 | `SimpleEnv.step` — `src/ppo_env.py` |
  | 182 | 13 | 30 | `Clustering.perform_kmeans` — `src/clustering.py` |
  | 72 | 8 | 25 | `Clustering.perform_HA` — `src/clustering.py` |
  | 56 | 7 | 47 | `plot_result` — `src/features.py` |
  | 56 | 7 | 31 | `Invest_Table.ls_table` — `src/invest_table.py` |
  | 56 | 7 | 19 | `metrics.cal_MDD` — `src/metrics.py` |
  | 56 | 7 | 20 | `metrics.cal_calmar_ratio` — `src/metrics.py` |

  **There are no tests in this repository, so these are not real CRAP scores** — they are the upper
  bound each function would score at 0% coverage. Complexity is the only half of the measure that
  can be computed here.
- **The duplication the ranking points at.** `product_LS_Table` reaches cc 13 by holding three
  near-identical 25-line blocks (`save == 'LS'`, `'L'`, `'S'`) that differ only in how they mask and
  halve the position count. The same shape repeats one level up: `1_perform_clustering.py` is three
  near-identical blocks and `2_invest.py` is seven, which is why two flat scripts account for 300
  of the 830 NLOC and contain zero functions. Collapsing either would be a real improvement and a
  real risk: the blocks have quietly drifted apart (for example only the `Reversal` and
  `cointegration` blocks of `2_invest.py` skip months that already exist in the output directory),
  so a merge has to decide which behaviour is correct, and nothing here can be re-run to check.
  That is why this pass renamed and documented rather than merged.
- **Missing code, not missing data.** The contrastive-learning encoder that produced
  `data/Clustering_Result/Contrastive_Learning*/` is not in this repository, and neither are its
  per-firm probabilities (`.../Prob/`, `.gitignore`d) that `cl_outlier_filter.ipynb` needs. The
  committed cluster labels can be used, but that branch cannot be re-derived from this repository.
  The same holds for `data/characteristics_UK/`: the panel is committed and the UK sub-periods are
  written out in `4_analyze_portfolio.ipynb`, but no code path in this repository reads it.
