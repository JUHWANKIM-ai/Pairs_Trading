# Pairs trading on clustered firm characteristics (US equities, 1990-2022)

[![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)](#get-started)
[![pandas](https://img.shields.io/badge/pandas-%3C3.0-150458?logo=pandas&logoColor=white)](requirements.txt)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%E2%89%A51.2-F7931E?logo=scikitlearn&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-PPO%20side%20branch-EE4C2C?logo=pytorch&logoColor=white)](ppo_threshold.ipynb)
[![Data](https://img.shields.io/badge/data-WRDS%20Compustat%20%2B%20CRSP-1f6feb)](#2-data)
[![Backtest](https://img.shields.io/badge/backtest-391%20months%20(1990--02%20%E2%86%92%202022--08)-success)](#results)

This repository implements and extends the strategy of *Pairs trading via unsupervised learning*
(Chulwoo Han, Zhaodong He, Alenson Jun Wei Toh, **European Journal of Operational Research** 307,
2023, 929-947). Every month, the cross section of US firms is clustered on firm characteristics
rather than on return series; within each cluster, firms are paired best-against-worst by last
month's return, and the pairs whose return gap is wide enough are traded as a short-term reversal
(long the loser, short the winner) for one month. The clustering is where the alternatives live:
k-means, DBSCAN and agglomerative clustering are run over a grid of hyper-parameters, and a
contrastive-learning encoder is included through its committed cluster labels. Classical
cointegration pairs trading and a plain decile reversal are kept as benchmarks.

The backtest covers 391 monthly rebalances, 1990-02 to 2022-08. The monthly cross section holds
1,845 to 3,609 firms (2,755 on average); after the outlier bucket and the spread filter, 285 of them
are traded in an average month, 10.4% of the cross section (measured from
`results/Individual_Result/K_mean/50.csv`).

Inputs are the firm-characteristic panel produced by the companion repository
[`WRDS`](https://github.com/JUHWANKIM-ai/WRDS) (WRDS Compustat and CRSP, 78 characteristics plus 49
momentum columns per firm-month); that repository is the data stage for this one, and this one is
the strategy and backtest stage. A UK panel is committed as well, but no code path here reads it.

Everything is CSV. The repository holds **42,441 files and about 5.3 GiB** of them, because the unit
of data is one month of one configuration: 392 monthly files per configuration and 106
configurations of cluster labels. Nine Python files and five notebooks, 985 lines of code by
`lizard`'s count with 688 of them in `src/`, sit on top of that. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the module map, the conventions the code depends on, and
the complexity ranking.

## Pipeline

```
data/characteristics_US/YYYY-MM.csv            392 monthly cross sections, 1990-01 .. 2022-08
  │   firms x (78 characteristics + mom1..mom49)
  │
  ├── 1_perform_clustering.py ─────────────────────────────────────────────────────┐
  │     drop mom49, standardise, PCA to 99% explained variance, keep mom1          │
  │     then one of                                                                │
  │       k-means      K in {10,20,30,40,50,100}, outliers = points farther from   │
  │                    their centroid than the alpha-percentile neighbour distance │
  │       DBSCAN       minPts = ln(n), eps = percentile of mean kNN distance       │
  │       agglomerative  average linkage, distance threshold = percentile          │
  │     cluster 0 is always the outlier bucket                                     │
  │                                                                                ▼
  │                                      data/Clustering_Result/<family>/<config>/YYYY-MM.csv
  │                                                   firms, mom1, clusters
  │   (contrastive-learning labels are committed here too; their encoder is not in this repo.
  │    cl_outlier_filter.ipynb re-labels low-confidence firms as cluster 0 -> CL_pre_0.1..0.9)
  │                                                                                │
  ├── 2_invest.py ◄───────────────────────────────────────────────────────────────── 
  │     within each cluster, sort by mom1 and pair highest with lowest;
  │     spread = own mom1 - mirror mom1;  trade the pair if |spread| > std(spread);
  │     long the low-momentum leg, short the high-momentum leg;  cluster 0 never trades
  │     alternatives in the same script: cointegration (Engle-Granger p<0.01, ADF<0.05,
  │     KPSS>0.05, |z-spread|>2) and decile reversal
  │                                                                                ▼
  │                                          data/LS_Result/<family>/<config>/YYYY-MM.csv
  │                                     Firm Name, Momentum_1, Long Short, Long, Short, Cluster
  │                                               (generated; not committed)
  │                                                                                │
  ├── 0_preprocessing.ipynb ──► data/Momentum1_Winsorized.csv                      │
  │     realised next-month returns, 9,749 firms x 392 months; anything above 1.0 is set to 1.5
  │     and anything below -0.95 to -0.95 (the scipy winsorize in the same notebook is dead code)
  │                                                                                │
  ├── 3_check_performance.ipynb ◄────────────────────────────────────────────────────
  │     shift positions one month forward, multiply by the realised return,
  │     cap each position at -0.30, charge -0.5 where a held firm has no return
  │     (delisting proxy), equal-weight, take logs, optional per-position fee
  │                                                                                ▼
  │                              results/Total_Result/Result/<family>_result_modified.csv   (391 monthly returns)
  │                              results/Total_Result/Result/<family>_statistcs_modified.csv (performance table)
  │                                                                                │
  └── 4_analyze_portfolio.ipynb ◄───────────────────────────────────────────────────
        the same metrics recomputed over seven sub-periods (dot-com, GFC, Covid, ...)
                                                                                   ▼
                                                       results/Total_Result/period_analysis.csv

side branch: ppo_threshold.ipynb
  a PPO agent observes (market mean return, market std, spread mean, spread std) and picks the
  spread threshold multiplier from {0.5, 1.0, 1.5} instead of the fixed 1.0;
  trained on 1990-1999, evaluated 2000-2022 on data/Clustering_Result/Best/
```

## Results

The headline table is `results/Total_Result/Result/Best_statistcs_modified.csv`, read back here
unchanged. It holds one configuration per clustering family — the four trees committed under
`data/Clustering_Result/Best/`, which are byte-identical to `Agglomerative/0.6`, `DBSCAN/0.1`,
`K_mean/50` and `Contrastive_Learning2/CL_pre_0.8/40_characteristics_us` — against the decile
reversal benchmark and the S&P 500 (`data/ETC/sp500_return_div.csv`). 391 monthly rebalances,
1990-02 to 2022-08, long/short, equal-weighted, and **gross of transaction costs**: the committed
run passes `fee = 0.000`.

| | Agglomerative | Contrastive learning | DBSCAN | k-means | Reversal | S&P 500 |
|---|---|---|---|---|---|---|
| Annualised return | 14.48% | 25.31% | 12.58% | 12.77% | 6.14% | 10.03% |
| Annualised std | 9.52% | 16.54% | 12.97% | 8.67% | 8.82% | 17.20% |
| Sharpe ratio | 1.52 | 1.53 | 0.97 | 1.47 | 0.70 | 0.58 |
| Sortino ratio | 2.10 | 3.17 | 1.43 | 2.55 | 1.40 | 0.73 |
| Profit factor | 3.73 | 3.72 | 2.57 | 3.42 | 1.84 | 1.59 |
| Maximum drawdown | -13.00% | -22.52% | -18.02% | -11.89% | -14.99% | -53.11% |
| Calmar ratio | 1.11 | 1.12 | 0.70 | 1.07 | 0.41 | 0.19 |
| Profitable years | 30 | 30 | 25 | 29 | 24 | 26 |
| Unprofitable years | 3 | 3 | 8 | 4 | 9 | 7 |
| Cumulative return | 8,091% | 155,802% | 4,655% | 4,914% | 596% | 2,154% |
| Mean monthly log return | 1.13% | 1.88% | 0.99% | 1.00% | 0.50% | 0.80% |
| t-statistic (Newey-West, 5 lags) | 8.38 | 8.06 | 5.16 | 7.75 | 4.33 | 3.49 |

Read the ratios against the definitions in `src/metrics.py`, not against the textbook ones: this
code annualises as `exp(12·mean)-1` and `exp(sqrt(12)·std)-1` and reports Sharpe as the ratio of the
two, with **no risk-free rate**. The t-statistic row is the Newey-West t of the mean monthly log
return against zero. `*_statistcs_modified.csv` also carries a second `t-statistic` row that tests
each column against the *second-to-last* column of its own table, which differs from file to file;
it is not reproduced here.

Per-hyper-parameter tables for each family are committed next to this one
(`K_mean_statistcs_modified.csv`, `DBSCAN_…`, `Agglomerative_…`, `Contrastive_Learning_…`,
`Contrastive_Learning2_…`), as are the long-only and short-only legs
(`results/Total_Result/{L,S}_Result/`) and the sub-period breakdown
(`results/Total_Result/period_analysis.csv`).

### Reinforcement-learning side branch

`ppo_threshold.ipynb` replaces the fixed spread threshold with a PPO agent choosing from
{0.5, 1.0, 1.5}, trained on 1990-1999 and evaluated on the rest. The saved outputs, on the
evaluation window only and therefore not comparable with the table above:

| Evaluated 2000-2022 | Agglomerative | Contrastive learning | DBSCAN | k-means |
|---|---|---|---|---|
| Annualised return, fixed threshold | 11.56% | 23.70% | 8.84% | 11.17% |
| Annualised return, PPO threshold | 13.57% | 24.01% | 11.10% | 10.98% |
| Sharpe ratio, fixed threshold | 1.18 | 1.27 | 0.70 | 1.21 |
| Sharpe ratio, PPO threshold | 1.23 | 1.23 | 1.08 | 1.12 |

Treat these as indicative, not as a result: the "fixed threshold" numbers come from a variable
(`return_dict`) that the saved notebook never defines — the cell that produced it was edited into
the PPO version and re-run, so only the PPO leg is reproducible from the committed code. The
figures in `results/images/` were saved from this notebook.

## Repository layout

| Path | Content |
|---|---|
| `0_preprocessing.ipynb` | winsorises the realised-return matrix into `data/Momentum1_Winsorized.csv` |
| `1_perform_clustering.py` | PCA + clustering per month; blocks enabled by the `K_mean_Save` / `dbscan_Save` / `agglomerative_Save` flags |
| `2_invest.py` | cluster labels to long/short tables; blocks for each clustering family plus `Reversal_Save`, `cointegration`, `Best` |
| `3_check_performance.ipynb` | long/short tables to monthly return series and the performance table; `save` selects LS / long-only / short-only |
| `4_analyze_portfolio.ipynb` | the same metrics per sub-period (US and UK period lists) |
| `ppo_threshold.ipynb` | PPO agent that picks the spread threshold; includes a CartPole reference loop |
| `cl_outlier_filter.ipynb` | re-labels the least confident firms of a contrastive-learning run as cluster 0 |
| `src/features.py` | the shared base: monthly CSV reading, momentum-prefix sniffing, PCA, long/short merge, return series, plotting and saving |
| `src/clustering.py` | `Clustering`: k-means, DBSCAN, agglomerative, each with its outlier rule |
| `src/invest_table.py` | `Invest_Table`: cluster labels to a long/short table, and the decile-reversal benchmark |
| `src/cointegration.py` | `cointegration`: Engle-Granger / ADF / KPSS pair screening, parallelised |
| `src/metrics.py` | `metrics`: the performance table |
| `src/ppo_env.py` | `SimpleEnv`, the `gym.Env` wrapper used by `ppo_threshold.ipynb` |
| `data/characteristics_US/` | 392 monthly cross sections, 1990-01 to 2022-08, 78 characteristics + `mom1..mom49` |
| `data/characteristics_UK/` | 402 monthly cross sections, 1990-02 to 2023-07, 10 characteristics + `Mom_1..Mom_49`; committed but unused by the code here |
| `data/Clustering_Result/` | 41,552 files: cluster labels, one CSV per month for each of 106 configurations |
| `data/Clustering_Result/Best/` | the one configuration per family that the headline table and the PPO notebook use |
| `data/Momentum1_{Original,Winsorized}.csv` | realised next-month returns, 9,749 firms x 392 months |
| `data/ETC/` | benchmarks and raw return matrices (`sp500_return_div.csv`, `FTSE_return.csv`, `momentum_us.csv`, ...) |
| `data/LS_Result/` | generated: the monthly long/short tables; `.gitignore`d, regenerate with `2_invest.py` |
| `results/Total_Result/` | monthly return series, performance tables, sub-period breakdown |
| `results/Individual_Result/` | per-month clustering diagnostics: firms traded, outliers, two largest clusters, cluster count |
| `results/images/` | the eight PNGs from `ppo_threshold.ipynb` |
| `ARCHITECTURE.md` | module map, dependency graph, conventions, complexity ranking |
| `requirements.txt` | dependencies, with the version bounds the code forces |

## Get Started

**1. Install.** Nothing in the code pins a Python version; it uses f-strings and type hints only.

```bash
pip install -r requirements.txt
```

`torch` and `gym` are needed only for `ppo_threshold.ipynb`. `pandas` must be older than 3.0 —
`DataFrame.applymap` and `fillna(method='ffill')` were removed there — and `scikit-learn` 1.2 or
newer, because `AgglomerativeClustering(metric=...)` was called `affinity` before that.

**2. Data.** The characteristic panels, the return matrices, the benchmarks and all 106
configurations of cluster labels are committed, so the repository is self-contained for everything
except two trees:

- `data/LS_Result/` — the monthly long/short tables, `.gitignore`d. Regenerate them with
  `2_invest.py` before running `3_check_performance.ipynb`.
- `data/Clustering_Result/Contrastive_Learning*/Prob/` — the per-firm cluster probabilities,
  `.gitignore`d. `cl_outlier_filter.ipynb` needs them, and the encoder that produces them is not in
  this repository; its cluster labels are committed instead.

To rebuild `data/characteristics_US/` from scratch, use [`WRDS`](https://github.com/JUHWANKIM-ai/WRDS)
— it needs a WRDS subscription for Compustat and CRSP.

**3. Run.** Every path in the code is relative to the repository root, so run from there. Nothing
creates its own output directory: make `data/LS_Result/<family>/<config>/` before a run that writes
there.

```bash
python 1_perform_clustering.py    # -> data/Clustering_Result/K_mean/40/
python 2_invest.py                # -> data/LS_Result/K_mean/*/ and results/Individual_Result/K_mean/
jupyter notebook                  # 0_, 3_, 4_ and the two side branches
```

Both scripts are driven by `bool` flags at the top of each block rather than by arguments. As
committed, only the k-means block of each is enabled, and `1_perform_clustering.py` has
`test_Set = [40]`, so a plain run reproduces one configuration rather than the whole grid; edit the
flags and `test_Set` to reproduce the rest. The exception is
`cl_outlier_filter.ipynb`, which expects to be run from a sub-directory (its paths start `../data/`).

## Related

[`WRDS`](https://github.com/JUHWANKIM-ai/WRDS) builds `data/characteristics_US/`: it takes Jeremiah
Green's stacked firm-characteristic file, keeps the 78 characteristics Han et al. use, splits it
into one CSV per month end and appends the momentum columns. That repository is the data stage and
stops before any model; this one is the strategy and backtest stage that consumes its output.
