# Seasonal Forecast MME Pipeline — Teaching Documentation

**System:** Seasonal_v2 Beta  
**Purpose:** A CPT-faithful Multi-Model Ensemble (MME) pipeline for seasonal probabilistic forecasting  
**Approach:** EOF/CCA with leave-one-out cross-validation, t-distribution probabilities, and RPSS skill scoring

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Core Concepts and Vocabulary](#2-core-concepts-and-vocabulary)
3. [Data and Directory Structure](#3-data-and-directory-structure)
4. [Pipeline Overview](#4-pipeline-overview)
5. [Step-by-Step Methodology](#5-step-by-step-methodology)
   - [Step 1: Data Loading](#step-1-data-loading)
   - [Step 2: Preprocessing](#step-2-preprocessing)
   - [Step 3: Standardization](#step-3-standardization)
   - [Step 4: EOF Decomposition](#step-4-eof-decomposition)
   - [Step 5: CCA Mode Optimization and Cross-Validation](#step-5-cca-mode-optimization-and-cross-validation)
   - [Step 6: Back-Projection to Physical Space](#step-6-back-projection-to-physical-space)
   - [Step 7: Error Variance Estimation](#step-7-error-variance-estimation)
   - [Step 8: Probabilistic Forecast Generation](#step-8-probabilistic-forecast-generation)
   - [Step 9: Skill Assessment (RPSS)](#step-9-skill-assessment-rpss)
   - [Step 10: Real-Time Forecast Generation](#step-10-real-time-forecast-generation)
6. [Parallelism and Performance](#6-parallelism-and-performance)
7. [Key Design Decisions and Why](#7-key-design-decisions-and-why)
8. [Configuration Reference](#8-configuration-reference)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Comparing Against the Benchmark](#10-comparing-against-the-benchmark)
11. [Glossary](#11-glossary)

---

## 1. What This System Does

This pipeline produces two types of output for a given season, variable (precipitation or temperature), and lead time:

1. **Hindcast skill scores** — How well did the system *would have* performed over a historical period (1991–2020)? Expressed as RPSS (Ranked Probability Skill Score), where 0 = no better than climatology, 1 = perfect, negative = worse than climatology.

2. **Real-time probabilistic forecasts** — For the *current* forecast, what is the probability that the seasonal total falls in the below-normal, normal, or above-normal tercile?

The system follows the methodology of **CPT (Climate Predictability Tool)** developed at the International Research Institute for Climate and Society (IRI/Columbia), extended to:
- Work with **multiple models** (up to 10 GCMs) in an ensemble
- Automate **mode selection** (rather than requiring manual tuning)
- Run **in parallel** across all 72 combinations (2 variables × 12 seasons × 3 leads)

---

## 2. Core Concepts and Vocabulary

| Term | Meaning |
|------|---------|
| **GCM** | General Circulation Model — a seasonal forecast model from a weather center |
| **Hindcast** | A retroactive forecast run over a historical period using the same method as real forecasts |
| **LOO / LOO-CV** | Leave-One-Out Cross-Validation — train on all years except the target year, predict the target |
| **EOF** | Empirical Orthogonal Function — a spatial pattern capturing dominant variance (essentially PCA on 2D fields) |
| **PC** | Principal Component — the time series of how strongly an EOF pattern is expressed each year |
| **CCA** | Canonical Correlation Analysis — finds linear combinations of two PC sets that are maximally correlated |
| **MME** | Multi-Model Ensemble — combining forecasts from several models |
| **RPSS** | Ranked Probability Skill Score — how much better is the forecast than climatology? |
| **Tercile** | Dividing the historical distribution into thirds: below-normal (<33rd percentile), normal, above-normal (>67th) |
| **Lead time** | How far in advance the forecast is issued (LEAD1 = shortest, LEAD3 = longest) |
| **t-distribution** | A probability distribution with heavier tails than Gaussian; appropriate for small samples |

---

## 3. Data and Directory Structure

### Input (pycpt directory)
The pycpt directory was produced by running the Climate Predictability Tool (CPT) operationally. Our system reads its input data directly:

```
pycpt/
├── {SEASON}{YEAR}_{VAR}_KMME/
│   └── 60W-30E_to_5S-35N/data/
│       ├── KAUST.PRCP.nc             ← observations
│       ├── CanSIPSIC4.PRCP.nc        ← model hindcast
│       ├── CFSv2.PRCP.nc
│       ├── ...
│       ├── CanSIPSIC4.PRCP_f2025.nc  ← model real-time forecast
│       └── ...
├── 2.LEAD2/
│   └── (same structure)
└── 3.LEAD3/
    └── (same structure)
```

Where:
- `SEASON` = 3-letter season abbreviation (e.g., `MAM`, `JJA`, `DJF`)
- `YEAR` = initialization year (e.g., `1991`)
- `VAR` = `PRCP` or `T2M`

### Output (new_system directory)
```
new_system/
└── {LEAD}/
    └── {SEASON}{YEAR}_{VAR}/
        ├── MME_skill_scores.nc    ← RPSS per grid point
        └── MME_forecast.nc        ← probabilistic forecast
```

---

## 4. Pipeline Overview

```
GCM Hindcast Files (10 models)
+ Observations (KAUST)
          │
          ▼
  [Step 1: Load]
  Read NetCDF files, detect and apply METEOFRANCE8↔9 fallbacks
          │
          ▼
  [Step 2: Preprocess]
  Temporal alignment → Regrid to obs grid → Mask bad points → Flatten to (T × space)
          │
          ▼
  [Step 3: Standardize]
  Per-grid-point zero-mean, unit-variance transformation
          │
          ▼
  [Step 4: EOF Decomposition]
  Observations: 6 modes  │  Each model: 8 modes
  Extract PCs for time, EOFs for space
          │
          ▼
  [Step 5: CCA Mode Optimization + LOO Cross-Validation]
  ┌─────────────────────────────────────────────────────┐
  │  For each model (in parallel):                      │
  │    For each (n_x, n_y, n_cca) combination:         │
  │      Run full LOO cross-validation                  │
  │      Score with mean domain RPSS                    │
  │    Select best modes                                │
  │    Refit CCA on full period with best modes         │
  └─────────────────────────────────────────────────────┘
          │
          ▼
  [Step 6: Back-project to Physical Space]
  Reconstruct LOO predictions from PC space → grid space
          │
          ▼
  [Step 7: Error Variance Estimation]
  Compute LOO residual variance per grid point per model
  Determine degrees of freedom: df = T - n_cca - 1
          │
          ▼
  [Step 8: Probabilistic Hindcast]
  LOO tercile thresholds (excludes target year ±3 window)
  P(Y ≤ threshold) via Student's t-CDF
  Average across models (equal weights)
          │
          ▼
  [Step 9: RPSS Skill Score]
  Compare tercile probabilities against climatological reference
  Save per-grid-point RPSS → MME_skill_scores.nc
          │                          │
          ▼                          ▼
  [Step 10: Real-Time Forecast]
  Load _f{year}.nc files        pycpt/ benchmark
  Project → CCA → Reconstruct   compare_all.py
  Tercile probabilities          comparison plots
  Save → MME_forecast.nc
```

---

## 5. Step-by-Step Methodology

### Step 1: Data Loading

**File:** `data_io.py`

#### What happens
- Observations are loaded from `KAUST.{VAR}.nc`
- Up to 10 GCM hindcast files are loaded, one per model
- Variables are detected flexibly (e.g., `aprod`, `prec`, `PRCP` all map to precipitation)
- Missing values (-999) are replaced with NaN
- Precipitation is clipped to ≥ 0 (negative precipitation is physically meaningless)

#### Key design decision: METEOFRANCE fallback
METEOFRANCE8 and METEOFRANCE9 are two versions of the same model. Some seasons only have one version available. The system tries the primary file first; if missing, it loads the alternative:

```
Try METEOFRANCE8.PRCP.nc → if missing, try METEOFRANCE9.PRCP.nc
Try METEOFRANCE9.PRCP.nc → if missing, try METEOFRANCE8.PRCP.nc
```

**Why:** Rather than losing a model entirely due to a file naming issue, we recover gracefully. Over 72 combinations, losing a model for a handful of seasons would silently reduce MME quality.

#### Teaching note
In production forecasting, data availability is never guaranteed. Building fallback logic into the loader prevents silent failures. Always log which fallback was used so you can audit it later.

---

### Step 2: Preprocessing

**File:** `preprocess.py`

This is the most important data quality step. A sequence of operations transforms raw NetCDF fields into a clean matrix ready for EOF/CCA.

#### 2a. Temporal alignment
Find the intersection of years across all models and observations, then trim to the configured hindcast window (1991–2020).

**Why:** Models may have different availability. CCA requires all inputs to share the same time dimension.

#### 2b. Regridding
Each GCM output is bilinearly interpolated onto the observation grid.

**Why:** GCMs have different native resolutions. EOF and CCA require all predictors and predictands to live on a common grid.

#### 2c. Sparse model filtering
Any model where fewer than 10% of grid points have valid data is dropped entirely for this combination.

```python
valid_fraction = np.isfinite(data).mean()
if valid_fraction < 0.10:
    drop model
```

**Why this matters:** METEOFRANCE9 can produce all-NaN fields for very dry seasons (where the model has no meaningful precipitation signal). Including such a model in CCA would inject pure noise — or worse, cause numerical failures.

**Teaching note:** Always check for models that are spatially mostly missing before statistical analysis. This is easy to miss and causes subtle degradation.

#### 2d. Masking
Three types of masking are applied:
- **Always-NaN:** Grid points with no data in any year are removed
- **Zero-variance:** Grid points where observations never change (std ≈ 0) are removed — CCA cannot extract signal from constants
- **Dry masking (optional):** If `drymask_threshold` is set, grid points with mean precipitation below the threshold are excluded (e.g., desert pixels that are always near-zero)

#### 2e. Flattening
The (lat, lon) spatial dimensions are collapsed into a single `space` dimension: shape becomes `(T, space)`.

**Why:** Linear algebra (SVD, CCA) operates on 2D matrices. Flattening the spatial dimensions is the standard approach. The (lat, lon) indices are saved and restored at the end.

---

### Step 3: Standardization

**File:** `transform.py`

Each grid point is transformed to zero mean and unit variance:

```
x_std = (x - mean(x)) / std(x)
```

The `std` uses Bessel's correction (ddof=1) and is clipped to a minimum of 1e-8 to prevent division-by-zero at near-constant points.

**Why this is required:**
- EOF decomposition is variance-based. Without standardization, grid points with large absolute values (e.g., tropical precipitation) dominate the leading EOFs, regardless of whether they carry predictable signal.
- Standardization ensures every grid point contributes equally to the decomposition.

**Key subtlety:** The standardization statistics (mean, std) computed on the training period (hindcast years) must also be applied to the forecast data during prediction. Applying different statistics to training and test data is one of the most common data leakage errors in climate forecasting.

---

### Step 4: EOF Decomposition

**File:** `eof.py`

EOF decomposition reduces dimensionality before CCA. It is applied separately to:
- The **observation field** (predictand): retain 6 modes
- Each **GCM field** (predictor): retain 8 modes per model

#### Algorithm
Thin Singular Value Decomposition:
```
U, s, Vt = SVD(X)          X shape: (T, space)
PC = (U × s)[:, :n]        shape: (T, n)  — time scores
EOF = Vt[:n, :]            shape: (n, space) — spatial patterns
```

The PCs are passed to CCA; the EOFs are stored for back-projection.

#### Why fixed mode counts (not variance-based)
A common alternative is to retain modes that together explain 70–80% of variance. This system uses **fixed truncation** instead.

**Rationale:**
1. **Signal filtering:** For precipitation, leading EOFs often capture large-scale patterns (ITCZ, monsoon) that are predictable by GCMs, while higher modes capture local noise. A fixed small number of modes acts as a deliberate low-pass filter.
2. **Consistency:** Across 72 combinations, variance thresholds would produce different numbers of modes for each case, making interpretation difficult.
3. **CPT convention:** This matches the approach used in CPT, which is the benchmark we are comparing against.

**Teaching note:** EOF mode selection is a hyperparameter choice with real consequences. Too few modes: lose predictable signal. Too many modes: inject unpredictable noise into CCA. The right number depends on the domain size, variable, and season.

---

### Step 5: CCA Mode Optimization and Cross-Validation

**File:** `cca.py`

This is the computational core of the system. It performs three tasks per model:
1. Exhaustive search for the optimal number of EOF modes and CCA components
2. Leave-one-out cross-validation to assess out-of-sample skill
3. A final full-period fit with the optimal settings

#### 5a. The search space

For each model, three hyperparameters are optimized:
- `n_x` ∈ [1 .. x_eof_modes=8]: number of predictor EOF modes to pass into CCA
- `n_y` ∈ [1 .. y_eof_modes=6]: number of predictand EOF modes to pass into CCA
- `n_cca` ∈ [1 .. cca_modes=3]: number of canonical components to retain

**Constraint:** `n_cca ≤ min(n_x, n_y)` (fundamental algebraic requirement of CCA)

Total combinations evaluated: roughly 40 per model (not all 8×6×3 are valid due to constraint).

#### 5b. Leave-one-out cross-validation (LOO-CV)

For each hyperparameter combination `(n_x, n_y, n_cca)`, a full LOO is run:

```
For each year t in [1991..2020]:
    Train = all years except [t-w .. t+w]   (w = crossvalidation_window = 3)
    
    1. Standardize train data per grid point (using train statistics only)
    2. Fit CCA on train PCs (X_train[:, :n_x] vs Y_train[:, :n_y])
    3. Apply CCA to test predictor: X_test[:, :n_x] → Y_pred_PC
    4. Apply variance inflation to Y_pred_PC
    5. Store Y_pred_PC[t] for scoring
```

Note the **exclusion window of ±3 years**. Year t and its 3 neighbors on each side are excluded from training. This prevents the model from "peeking" at years near the target — important because seasonal climate has low-frequency variations that would otherwise inflate apparent skill.

#### 5c. Variance inflation

CCA systematically underestimates the variance of its predictions (a known property of linear regression-based methods). The Barnett & Preisendorfer (1987) correction is applied:

```
inflation_factor = std(Y_true) / std(Y_pred)
Y_pred_inflated = Y_pred × inflation_factor
```

This restores the spread of the predictive distribution to match the observed spread, which is critical for calibrated probabilistic forecasts.

#### 5d. Scoring and mode selection

Each LOO run is scored using **RPSS in PC space** (fast approximate version):
```
Score = mean RPSS across the Y PC dimensions
```

The hyperparameter combination with the highest mean score is selected.

**Why RPSS, not RMSE or correlation?**
- The final output is probabilistic (tercile probabilities). RPSS directly measures the quality of probabilistic forecasts.
- Correlation and RMSE measure deterministic accuracy, which can be misleading for probability forecasts.
- Scoring in PC space is much faster than scoring in full grid space (O(n_cca) vs O(space)), allowing exhaustive search to remain practical.

#### 5e. Full-period refit

After the optimal `(n_x, n_y, n_cca)` is identified, CCA is refit on **all training years** using those settings. This final model is what generates the hindcast reconstructions and real-time forecasts.

#### 5f. Parallelism per model

Each model's search is independent. The searches are distributed across CPU cores using `ProcessPoolExecutor`. This is the dominant computational cost: ~15–20 seconds per combination (wall time), most of which is CCA fitting.

**Teaching note:** The mode search is an outer hyperparameter loop wrapping an inner LOO. It is computationally expensive but necessary — CCA performance is sensitive to truncation choice, and there is no analytical solution that avoids search.

---

### Step 6: Back-Projection to Physical Space

**File:** `main.py` (within `run_hindcast`)

The LOO predictions from Step 5 are in PC space (shape `(T, n_y)`). They must be reconstructed to the physical observation grid `(T, space)` before computing spatial skill metrics.

```
Y_pred_grid = Y_pred_PC @ EOF_y[:n_y, :]
```

This is the inverse of the EOF projection: multiply PC time series by the spatial EOF patterns to get gridded predictions.

The same operation is applied in the forecast stage (Step 10).

---

### Step 7: Error Variance Estimation

**File:** `probabilistic.py`, function `compute_error_variance`

After back-projection, we have both the LOO predictions `(Y_pred_grid)` and the true observations `(Y_true)` at each grid point for all T years. The LOO residuals are:

```
residuals[t, s] = Y_true[t, s] - Y_pred_grid[t, s]
error_variance[s] = var(residuals[:, s])           (per grid point)
df = T - n_cca - 1                                 (degrees of freedom)
```

The **degrees of freedom** formula `T - n_cca - 1` follows the CPT convention for a regression with `n_cca` canonical predictors. It accounts for the fact that each additional CCA component uses up one degree of freedom from the residuals.

**Why this matters:** The error variance and df together parametrize the t-distribution used in Step 8. If you underestimate error variance, probabilities will be overconfident (too narrow). If you use too many df, the distribution is too Gaussian and underestimates tail probabilities.

---

### Step 8: Probabilistic Forecast Generation

**File:** `probabilistic.py`

This step converts the deterministic reconstruction into calibrated probability forecasts.

#### The predictive distribution

For each grid point `s` and year `t`, the forecast is treated as a **Student's t-distribution**:

```
Y ~ t(location = mu, scale = sigma, df = df)
where:
    mu = Y_pred_grid[t, s]            (best estimate)
    sigma = sqrt(error_variance[s])   (estimated uncertainty)
    df = T - n_cca - 1               (degrees of freedom)
```

#### LOO-aware tercile thresholds

The climatological tercile boundaries (33rd and 67th percentile) must be computed carefully:

```
For each year t:
    train_years = all years except [t-w .. t+w]
    threshold_33[t, s] = percentile(Y_true[train_years, s], 33)
    threshold_67[t, s] = percentile(Y_true[train_years, s], 67)
```

**Why LOO thresholds?** If we used the full-period thresholds for all years, the observations for year t would influence the threshold used to categorize year t. This is a form of data leakage that inflates apparent skill. LOO thresholds ensure the categorization of year t is based only on other years — the true out-of-sample standard.

#### Probability computation

```
P(below) = t_CDF(threshold_33, mu, sigma, df)
P(above) = 1 - t_CDF(threshold_67, mu, sigma, df)
P(normal) = 1 - P(below) - P(above)
```

#### Multi-model ensemble averaging

Each model produces its own probability forecast. The MME is the simple average:

```
P_MME = mean over models of P_model
```

**Why equal weights?** Weighted averaging (e.g., by historical skill) is more complex and requires careful estimation to avoid overfitting the weights to the training period. Studies show equal-weight MME is often competitive with skill-weighted MME, especially when the number of models is moderate and the historical record is short. It is also more transparent and reproducible.

---

### Step 9: Skill Assessment (RPSS)

**File:** `skill.py`

RPSS is the standard metric for probabilistic categorical forecasts in seasonal climate prediction.

#### How RPSS is computed

For each grid point, across all T hindcast years:

1. **Convert observations to one-hot format:**  
   Each year is assigned to a category: [1,0,0] (below), [0,1,0] (normal), or [0,0,1] (above)

2. **Compute Ranked Probability Score (RPS) for the forecast:**  
   ```
   fcst_cumulative = cumsum(P_below, P_normal, P_above)   → shape (T, 3)
   obs_cumulative  = cumsum(obs_onehot)                   → shape (T, 3)
   RPS_fcst = mean((fcst_cumulative - obs_cumulative)²)
   ```

3. **Compute RPS for climatological reference:**  
   Climatology always issues 1/3 probability to each category:
   ```
   clim_cumulative = [1/3, 2/3, 1.0] for every year
   RPS_clim = mean((clim_cumulative - obs_cumulative)²)
   ```

4. **RPSS = 1 - RPS_fcst / RPS_clim**

An RPSS of 0 means no better than climatology. Positive values indicate skill. Negative values indicate the forecast is worse than just saying "equal chance."

**Teaching note:** RPSS is always reported with respect to a reference (here, climatological equal-chance). The same forecast skill score can look very different depending on which reference you choose. Always specify the reference when reporting RPSS.

---

### Step 10: Real-Time Forecast Generation

**File:** `main.py`, function `run_forecast_stage`

The real-time forecast uses the **same trained CCA models** from the hindcast stage, but applied to the actual current-year model output.

#### Steps

1. **Load forecast files:** Files named `MODEL._f{fdate.year}.nc` (e.g., `CFSv2.PRCP_f2025.nc`). Models without a forecast file are silently skipped and logged.

2. **Preprocess:** Apply the same regridding and masking as the hindcast, using the same valid-point mask derived during hindcast (spatial consistency is critical).

3. **Standardize:** Apply the **hindcast-period** mean and std (not computed fresh from the forecast). This is essential: the forecast is a single time step, so you cannot compute meaningful statistics from it alone.

4. **EOF projection:** Project the forecast onto the **hindcast EOFs** (not re-computed). `PC_fcst = X_fcst @ EOF_x.T`

5. **CCA transform:** Apply the trained CCA transformation to get predictand PCs.

6. **Back-project:** Reconstruct to observation grid.

7. **Probability computation:** Use **full-period** tercile thresholds (since there is no need to exclude a LOO window for a real forecast).

8. **MME averaging:** Average across available models.

#### Output: `MME_forecast.nc`
- `probabilistic`: probabilities for below/normal/above tercile — shape (category=3, lat, lon)
- `prob_exceed_10` and `prob_exceed_90`: exceedance probabilities for extremes

---

## 6. Parallelism and Performance

The pipeline is designed to scale across a multi-core machine. There are two levels of parallelism:

### Level 1: Combination-level (run_all.py)
Multiple (lead × season × variable) combinations run simultaneously using `ProcessPoolExecutor`. The `--jobs N` flag controls how many combinations run in parallel.

### Level 2: Model-level (cca.py)
Within each combination, the CCA mode search for each model runs in a separate process. With 10 models and 40 hyperparameter combinations each, this is the dominant cost.

### Recommended settings
| Machine | CPU cores | `--jobs` | CCA workers per job |
|---------|-----------|----------|---------------------|
| Laptop  | 8         | 1        | 8                   |
| Workstation | 16   | 2        | 8                   |
| Server  | 52       | 5        | 10                  |

**Formula:** `jobs × cca_workers ≈ cpu_count`

The code uses `mp.get_context("spawn")` for the outer pool to avoid nested fork/spawn conflicts — a subtle but important implementation detail when nesting `ProcessPoolExecutor`.

### Estimated wall times (per combination)
| Step | Time |
|------|------|
| Load + preprocess | ~1s |
| Standardize + EOF | ~0.5s |
| CCA mode search (9 models, parallel) | ~12–15s |
| Probabilistic + skill | ~1s |
| Forecast stage | ~3s |
| **Total** | **~18–20s** |

With 72 combinations and `--jobs 5`, total wall time is approximately **5–7 minutes** on a 52-core server.

---

## 7. Key Design Decisions and Why

### Why not use a neural network or ML-based approach?

For seasonal climate forecasting over AP:
- **Small sample:** Only ~30 hindcast years are available. Neural networks typically need thousands of samples per spatial degree of freedom.
- **Interpretability:** Operational forecasters need to understand and trust the method. EOF/CCA spatial patterns are physically interpretable (teleconnections, ENSO footprints).
- **Benchmark compatibility:** CPT is the operational standard. Matching its methodology makes validation straightforward.

### Why CCA specifically?

CCA finds the linear combinations of predictor and predictand PCs that are **maximally correlated** — exactly what you want when asking "what combination of GCM signals best predicts observed rainfall patterns?" It explicitly links predictor variability to predictand variability, unlike simple regression of each grid point independently.

### Why per-model CCA (not a combined multi-model predictor)?

**Alternative approach:** Concatenate all models into one large predictor matrix, then run a single CCA.

**Problem with concatenation:** Different models have very different skill, noise characteristics, and teleconnection patterns. A model with high noise would dominate the combined EOFs, degrading the others.

**Our approach:** Each model runs CCA independently, then probabilities are averaged at the end. This lets the mode selection tune independently for each model's characteristics.

### Why LOO rather than a fixed split (training/validation)?

With only 30 years, a fixed 25-year training / 5-year validation split wastes data. LOO uses all 30 years for training and all 30 years for validation, producing the most efficient estimate of out-of-sample skill. The exclusion window (±3 years) guards against temporal autocorrelation inflating the skill estimate.

### Why Student's t instead of Gaussian?

With 30 hindcast years and 3 CCA components, the effective sample size for error estimation is `df = 30 - 3 - 1 = 26`. The Student's t-distribution with 26 df has somewhat heavier tails than a Gaussian. For extreme probability estimates (prob of >90th percentile), this difference matters — Gaussian would be overconfident.

### Why RPSS and not just correlation?

Forecasters and decision-makers ultimately need to know "what is the chance of below-normal rainfall?" — a question about probabilities, not correlations. RPSS directly measures how well the probability forecasts perform. It also penalizes overconfident forecasts (putting all probability on the wrong category is heavily penalized).

---

## 8. Configuration Reference

All settings live in `config.yaml` (or are injected by `run_all.py` in batch mode):

```yaml
# Forecast initialization date
fdate: 2025-02-01

# Hindcast period
target_first_year: 1991
target_final_year: 2020

# Season and variable
target: "Mar-May"          # Written as "Mon-Mon" or "3-letter month" format
variable: PRCP             # PRCP or T2M

# Models to include
models:
  - CanSIPSIC4
  - CFSv2
  - GEOSS2S
  - GLOSEA6
  - SEAS51c
  - SPSv3p5
  - CCSM4
  - GCFS2p1
  - METEOFRANCE8
  - SPEAR

# EOF/CCA hyperparameters
y_eof_modes: 6             # How many obs EOF modes to retain
x_eof_modes: 8             # How many predictor EOF modes per model
cca_modes: 3               # Maximum number of CCA components to consider

# Cross-validation
crossvalidation_window: 3  # Years excluded around LOO target (each side)

# Masking
drymask_threshold: null    # Optional: exclude very dry points (mm/month)

# Probabilistic output
percentiles: [10, 33, 67, 90]   # Which thresholds to compute probabilities for

# Plotting (disabled for batch runs)
plot:
  skill: false
  forecast: false
```

### Sensitivity guidance

| Parameter | If you increase | If you decrease |
|-----------|----------------|-----------------|
| `y_eof_modes` | More obs spatial detail in CCA; risk of fitting noise | More aggressive signal filtering |
| `x_eof_modes` | More predictor detail; larger search space | Faster search; may miss signal |
| `cca_modes` | More CCA pairs considered; risk of overfitting | Fewer pairs; faster |
| `crossvalidation_window` | Stricter LOO; less optimistic skill estimate | More optimistic skill estimate |
| `drymask_threshold` | Fewer valid points; better signal-to-noise in dry regions | All points included |

---

## 9. Running the Pipeline

### Single combination (for testing / debugging)
```bash
python main.py config.yaml
```

Edit `config.yaml` first to specify the combination.

### Batch run (recommended for production)
```bash
# Run all 72 combinations, 5 at a time
python run_all.py --jobs 5

# Dry run: see what would run without computing
python run_all.py --dry-run

# Only one lead or variable
python run_all.py --lead LEAD1 --var PRCP

# Force recompute even if output exists
python run_all.py --force --jobs 5
```

### Checking progress
Each completed combination prints a success/skip/fail status. Failures are logged with the exception.

---

## 10. Comparing Against the Benchmark

`compare_all.py` loads both the new-system RPSS and pycpt RPSS for every combination and produces:

1. **Summary scatter plot (`comparison_summary.png`):** One point per combination, new RPSS on Y axis, pycpt RPSS on X axis. Points above the diagonal = new system wins.

2. **Per-combination maps (`comparison_maps/*.png`, optional):** Three panels — pycpt RPSS map, new RPSS map, difference map.

```bash
python compare_all.py             # Summary only
python compare_all.py --maps      # Summary + individual maps
```

### Interpreting the comparison

- **Diagonal alignment:** If points cluster near the 1:1 diagonal, the two systems agree. This validates the new system is implementing the same methodology.
- **Systematic offset:** If points cluster above the diagonal, the new system has higher RPSS everywhere — possibly due to better mode selection or model handling.
- **Scatter around diagonal:** High scatter indicates the systems make different choices for specific combinations (season/lead/variable). Investigate the outlier combinations.

---

## 11. Glossary

| Term | Definition |
|------|-----------|
| **Backprojection** | Multiplying PC time series by EOF spatial patterns to recover gridded fields |
| **Canonical variates** | The transformed variables found by CCA that are maximally correlated across two datasets |
| **Cumulative distribution function (CDF)** | P(X ≤ x); used here to convert a t-distribution into a probability |
| **Degrees of freedom** | A parameter controlling the shape of the t-distribution; lower = heavier tails |
| **Hindcast** | A forecast made for a past period using the same method as real-time forecasts, used for skill estimation |
| **LOO cross-validation** | Training on N-1 samples and testing on the left-out sample, repeated N times |
| **MME** | Multi-Model Ensemble; combining predictions from multiple GCMs |
| **Mode truncation** | Keeping only the first k EOFs, discarding the rest |
| **RPSS** | Ranked Probability Skill Score; how much better than climatological equal-chance |
| **Sparse model** | A model where nearly all spatial grid points are missing; filtered out before CCA |
| **Standardization** | Subtracting mean and dividing by std so all variables are on the same scale |
| **Student's t-distribution** | A bell-shaped distribution with heavier tails than Gaussian; parameterized by location, scale, and df |
| **Tercile** | One of three equally likely categories based on historical climatology: below-normal, normal, above-normal |
| **Variance inflation** | Scaling CCA predictions to restore their variance to observed levels |

---

*This document describes the methodology as implemented in Seasonal_v2/Beta. For implementation details, refer to the source files listed in each section.*
