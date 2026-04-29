# Inferring Optimised Pregnancy Conception Dates from Multiple Gestational Age Estimates

**A Hybrid Anomaly Detection and Linear Mixed Model Approach**  
Chepkirui M et al. — *BMC Medical Informatics and Decision Making* (under review)

KEMRI Centre for Global Health Research, Kisumu, Kenya &  
Liverpool School of Tropical Medicine, Liverpool, United Kingdom

---

## Background

Accurate gestational age (GA) estimation is fundamental for perinatal research and pharmacovigilance but remains challenging in low-resource settings where routine pregnancy records are inconsistent, early ultrasound is limited, and multiple GA estimates often conflict. This challenge is most acute in large-scale observational studies and pharmacovigilance registries, where the data quality controls available in clinical trials are rarely feasible.

This repository contains the implementation code for a two-stage pipeline applied to data from the **MiMBa Pregnancy Exposure Registry**, spanning 42 routine healthcare facilities in Homa Bay County, western Kenya. The pipeline first applies Isolation Forest to identify and exclude implausible GA estimates, then uses a linear mixed-effects model to synthesise the remaining measurements into a single calibrated "best" estimate, accounting for systematic biases and pregnancy-level variation. The model pools information across all pregnancies to generate a calibrated conception date for each individual woman, even when her own measurements are limited.

For full results, please refer to the manuscript.

---

## Pipeline Overview

The pipeline follows a three-stage workflow, corresponding to the implementation steps described in the manuscript:

| Step | Stage | Script | Description |
|------|-------|--------|-------------|
| 1 | **Data consolidation** | `00_data_consolidation.R` | Extract and link pregnancy episode records across ANC, maternity, and EMR registers; engineer Isolation Forest features |
| 2 | **Anomaly detection** | `01_anomaly_detection_isolation_forest.py` | Identify and filter implausible GA observations using outcome-stratified Isolation Forest |
| 3 | **Statistical integration** | `02_lmm_ga_estimation.R` | Combine multiple GA measurements into a single calibrated estimate using a linear mixed-effects model, anchored to first-trimester ultrasound |

**Scripts must be run in order.** Each stage produces the input for the next.

```
Raw multi-register linked data (MiMBa Registry)
        │
        ▼
00_data_consolidation.R
  └─ Outcome filtering 
  └─ Wide-to-long reshape (1 row per pregnancy × method)
  └─ Physiological plausibility bounds per outcome
  └─ Feature engineering (ga_edd, ga_del, ga_zscore, ga_deviation, z_gestation)
  └─ Ultrasound trimester classification (US1/US2/US3)
        │
        ▼
all_records_for_pythonIsolationF.csv
        │
        ▼
01_anomaly_detection_isolation_forest.py
  └─ Outcome-stratified Isolation Forest (4 strata)
  └─ Hyperparameter tuning via silhouette-score grid search
  └─ 5-fold cross-validation
  └─ Permutation importance + Cohen's d feature analysis
        │
        ▼
clean_data_for_lmm.csv
        │
        ▼
02_lmm_ga_estimation.R
  └─ LMM: GA_birth ~ GA_method + birth_outcome + (1 | pregnancy_id)
  └─ BLUP-derived best-GA prediction anchored to US1
  └─ Bland-Altman bias assessment by method
  └─ Shrinkage visualisation (Figure 2 in manuscript)
        │
        ▼
Calibrated conception date per pregnancy (calibrated_cdate_data.rds)
```

---

## Repository Structure

```
mimba-ga-estimation/
├── README.md
├── requirements.txt                            # Python dependencies
├── 00_data_consolidation.R                     # Stage 0: feature engineering
├── 01_anomaly_detection_isolation_forest.py    # Stage 1: anomaly detection
├── 02_lmm_ga_estimation.R                      # Stage 2: LMM integration
├── data/
│   └── .gitkeep                                # data files are gitignored
└── outputs/
    └── .gitkeep                                # generated outputs gitignored
```

> **Note:** No patient data are included in this repository. The MiMBa Registry data are held under institutional data governance agreements at KEMRI. Researchers wishing to replicate this analysis should contact the corresponding author regarding data access.

---

## Installation

### R (Stages 0 and 2)

Requires R ≥ 4.1. Tested on R 4.4.3.

```r
install.packages(c(
  "tidyverse", "data.table", "janitor", "writexl", "openxlsx",
  "Hmisc", "merTools", "patchwork", "jtools", "lmerTest",
  "rlang", "ggplot2", "ggrepel", "cowplot", "showtext"
))
```

### Python (Stage 1)

Requires Python ≥ 3.8. Tested on Python 3.11 with scikit-learn 1.0.2.

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy
pandas
scikit-learn==1.0.2
scipy
matplotlib
seaborn
```

---

## Usage

### Stage 0 — Data Consolidation

```bash
Rscript 00_data_consolidation.R
```

### Stage 1 — Anomaly Detection

Update `INPUT_FILE` at the top of the script, then run:

```bash
python 01_anomaly_detection_isolation_forest.py
```

**Required input columns:**

| Column | Description |
|--------|-------------|
| `outcomef` | Pregnancy outcome (`Live_Birth_Term`, `Live_Birth_Preterm`, `Miscarriage`, `Stillbirth`) |
| `ga_edd` | GA at estimated date of delivery (weeks) |
| `ga_del` | GA at delivery (weeks) |
| `ga_zscore` | Within-pregnancy GA z-score |
| `ga_deviation` | Distance from outcome-specific expected GA range (weeks) |
| `z_gestation` | Outcome-level scaled gestation |
| `expected_ga_min` | Lower bound of expected GA range (weeks) |
| `expected_ga_max` | Upper bound of expected GA range (weeks) |

**Outputs:**

| File | Description |
|------|-------------|
| `all_data_with_anomaly_flags.csv` | All records with `is_anomaly` and `anomaly_score` columns added |
| `clean_data_for_bayesian.csv` | Anomaly-free records — **input to Stage 2** |
| `anomalies_for_review.csv` | Flagged records sorted by anomaly score for clinical review |
| `anomaly_detection_summary.csv` | Per-outcome model performance |


---

### Stage 2 — Linear Mixed-Effects Model

Update `INPUT_FILE` at the top of the script, then run:

```bash
Rscript 02_lmm_ga_estimation.R
```

**Model specification:**

```
GAᵢⱼ = β₀ + βGA_methodⱼ + βoutcomeₖ + uᵢ + εᵢⱼ
```

- `β₀` — intercept for first-trimester ultrasound (US1) and term live birth (reference)
- `βGA_methodⱼ` — fixed effects capturing systematic bias of each dating method relative to US1
- `βoutcomeₖ` — fixed effects for outcome category (Live Birth Term, Live Birth Preterm, Stillbirth, Miscarriage)
- `uᵢ` — pregnancy-specific random intercept; enables BLUP shrinkage, pooling information across all pregnancies
- `εᵢⱼ ~ N(0, σ²)` — residual error

In R notation: `GA_birth ~ GA_method + birth_outcome + (1 | pregnancy_id)`

**Outputs:**

| File | Description |
|------|-------------|
| `calibrated_cdate_data.rds` | Per-pregnancy best-GA predictions and consensus conception dates |


---

## Software

| Component | Version |
|-----------|---------|
| R | 4.4.3 |
| Python | 3.11 |
| scikit-learn | 1.0.2 |

---

## Citation

If you use this code in your research, please cite:

```
Chepkirui M, Dellicour S, Omondi B, Maube K, Ongayo G, Alaw M, Asuke B,
Ochieng T, Kariuki S, Oneko M, Barsosio H, ter Kuile F, Taegtmeyer M,
Lesosky M. Inferring Optimised Pregnancy Conception Dates from Multiple
Gestational Age Estimates: A Hybrid Anomaly Detection and Linear Mixed Model
Approach. BMC Medical Informatics and Decision Making (under review). 2026.
GitHub: https://github.com/mchepkirui/Gestation-Age-Estimation-Isolation-Forest-and-linear-mixed-effect-model
```

---

## Corresponding Author

Mercy Chepkirui  
[mercy.chepkirui@lstmed.ac.uk](mailto:mercy.chepkirui@lstmed.ac.uk)  
KEMRI Centre for Global Health Research / Liverpool School of Tropical Medicine
