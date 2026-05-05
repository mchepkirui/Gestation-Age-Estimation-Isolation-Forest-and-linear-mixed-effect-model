"""
Gestational Age Estimation Pipeline — Stage 1: Anomaly Detection

Description
-----------
This script implements Stage 1 (Anomaly Detection) of a two-stage pipeline
for estimating gestational age (GA) from routine multi-register pregnancy
data. It applies an outcome-stratified Isolation Forest to identify and
flag implausible GA measurements before statistical integration in Stage 2.


Workflow (this script)
----------------------
Step 1  Data ingestion        : Load the linked multi-register dataset.
Step 2  Outcome stratification: Partition records by pregnancy outcome, since
                                clinically plausible GA ranges differ by
                                outcome (e.g., miscarriage vs. term birth).
Step 3  Feature construction  : Select outcome-appropriate GA-derived features.
Step 4  Hyperparameter tuning : Grid search over contamination, n_estimators,
                                max_samples, and max_features; optimise using
                                silhouette score.
Step 5  Model training        : Fit final Isolation Forest per outcome stratum.
Step 6  Cross-validation      : 5-fold CV to assess anomaly-rate stability.
Step 7  Permutation importance: Quantify each feature's contribution to anomaly
                                detection by measuring prediction change on
                                random column permutation.
Step 8  Effect size analysis  : Cohen's d between anomalous and normal records
                                per feature, per outcome.
Step 9  Output export         : Save (a) full dataset with anomaly flags,
                                (b) clean dataset for Stage 2 modelling,
                                (c) anomaly-only dataset for clinical review.
                                

Outputs
-------
all_data_with_anomaly_flags.csv   — All records with is_anomaly, anomaly_score
clean_data_for_lmm.csv       — Anomaly-free records for LMM (Stage 2)
anomalies_for_review.csv          — Flagged records sorted by anomaly score
table1_model_performance.csv      — Manuscript Table 
table2_feature_importance.csv     — Manuscript Table 

Input
-----
all_records_for_pythonIF.csv
    Required columns:
        outcomef          — Pregnancy outcome category
                            (Live_Birth_Term | Live_Birth_Preterm |
                             Miscarriage | Stillbirth)
        ga_edd            — GA derived from estimated date of delivery 
        ga_del            — GA at delivery 
        ga_zscore         — Standardised GA z-score
        ga_deviation      — Deviation of GA from expected
        z_gestation       — Alternative gestation z-score
        expected_ga_min   — Lower bound of expected GA range
        expected_ga_max   — Upper bound of expected GA range 

Dependencies
------------
    numpy, pandas, scikit-learn, scipy, matplotlib, seaborn
    Install: pip install -r requirements.txt

Usage
-----
    python 01_anomaly_detection_isolation_forest.py

    The script is self-contained; no command-line arguments are required.
    Update INPUT_FILE below to point to the local data file.

"""

# =============================================================================
# 0. Imports
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 1. Configuration
# =============================================================================

INPUT_FILE = "all_records_for_pythonIF.csv"
RANDOM_STATE = 42

# Feature sets by outcome type.
# Pregnancy loss outcomes (miscarriage, stillbirth) lack an EDD-derived GA
# feature because these pregnancies often end before the EDD is established.
FEATURE_SETS = {
    "all_outcomes": [
        "ga_edd", "ga_del", "ga_zscore", "ga_deviation",
        "z_gestation", "expected_ga_min", "expected_ga_max",
    ],
    "pregnancy_loss": [          # miscarriage & stillbirth
        "ga_del", "ga_zscore", "ga_deviation",
        "z_gestation", "expected_ga_min", "expected_ga_max",
    ],
}

# Contamination search ranges are set from clinical domain knowledge:
# term births are expected to have very few implausible GA records, while
# miscarriage records are inherently more heterogeneous.
CONTAMINATION_RANGES = {
    "Live_Birth_Term":    [0.005, 0.010, 0.015, 0.020, 0.025],
    "Live_Birth_Preterm": [0.020, 0.025, 0.030, 0.035, 0.040],
    "Stillbirth":         [0.030, 0.035, 0.040, 0.045, 0.050],
    "Miscarriage":        [0.050, 0.055, 0.060, 0.065, 0.070],
}

# Human-readable labels for manuscript tables and plots
FEATURE_LABELS = {
    "ga_edd":          "GA at EDD",
    "ga_del":          "GA at Delivery",
    "ga_zscore":       "GA Z-score",
    "ga_deviation":    "GA Deviation",
    "z_gestation":     "Z-Gestation",
    "expected_ga_min": "Expected GA Min",
    "expected_ga_max": "Expected GA Max",
}

MIN_SAMPLES_PER_OUTCOME = 50   # skip strata with too few records
N_CV_FOLDS = 5
N_PERMUTATIONS = 30            # permutation importance replicates


# =============================================================================
# 2. Utility helpers
# =============================================================================

def safe_print(text):
    """Print text robustly, replacing characters that cannot be encoded."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(str(text).encode("utf-8", errors="replace").decode("utf-8"))


# =============================================================================
# 3. Data preparation
# =============================================================================

def prepare_outcome_specific_data(df, outcome_type):
    """
    Filter the dataset to a single outcome stratum and build its feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Full multi-outcome dataset.
    outcome_type : str
        One of the keys in CONTAMINATION_RANGES.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix for this outcome (NaN/inf replaced with column medians).
    features : list[str]
        Names of selected columns.
    outcome_data : pd.DataFrame
        Filtered rows for this outcome (preserves all original columns).
    """
    outcome_data = df[df["outcomef"] == outcome_type].copy()

    # Choose feature set: pregnancy-loss outcomes lack an EDD-based GA metric
    if outcome_type in ("Miscarriage", "Stillbirth"):
        features = FEATURE_SETS["pregnancy_loss"]
    else:
        features = FEATURE_SETS["all_outcomes"]

    X = outcome_data[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    return X, features, outcome_data


# =============================================================================
# 4. Hyperparameter tuning
# =============================================================================

def tune_isolation_forest(X, outcome_type, feature_names):
    """
    Select Isolation Forest hyperparameters via silhouette-score maximisation.

    A grid search is performed over n_estimators, max_samples, contamination,
    and max_features. For each candidate parameter set the silhouette score is
    computed on the resulting binary labels (inlier / outlier). The parameter
    set that maximises silhouette is retained.

    Silhouette score is used because there are no ground-truth anomaly labels
    in routine LMIC healthcare data; it provides an unsupervised measure of
    separation quality between the inlier and outlier clusters.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric, no NaNs).
    outcome_type : str
        Outcome stratum name (used to select contamination search range).
    feature_names : list[str]
        Column names (stored in results for traceability).

    Returns
    -------
    best_params : dict
        Best parameter set, with additional 'silhouette_score',
        'feature_names', and 'n_features' keys.
    scaler : StandardScaler
        Fitted scaler (must be applied to the same data before model.predict).
    """
    safe_print(f"\n=== Tuning {outcome_type} (n={len(X)}) ===")
    safe_print(f"  Features: {feature_names}")

    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_samples":   [256, 512, "auto"],
        "contamination": CONTAMINATION_RANGES[outcome_type],
        "max_features":  [1.0, 0.8],
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_score = -1
    best_params = None

    for params in ParameterGrid(param_grid):
        try:
            model = IsolationForest(**params, random_state=RANDOM_STATE)
            labels = model.fit_predict(X_scaled)

            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_params = {**params, "silhouette_score": score}
        except Exception:
            continue

    best_params["feature_names"] = feature_names
    best_params["n_features"] = len(feature_names)

    safe_print(f"  Best contamination : {best_params['contamination']:.3f}")
    safe_print(f"  Best n_estimators  : {best_params['n_estimators']}")
    safe_print(f"  Best max_samples   : {best_params['max_samples']}")
    safe_print(f"  Tuning silhouette  : {best_params['silhouette_score']:.3f}")

    return best_params, scaler


def _valid_model_params(best_params):
    """Return only the keys accepted by IsolationForest's constructor."""
    valid = {"n_estimators", "max_samples", "contamination", "max_features"}
    return {k: v for k, v in best_params.items() if k in valid}


# =============================================================================
# 5. Cross-validation
# =============================================================================

def cross_validate_model(model_params, X_scaled, n_folds=N_CV_FOLDS):
    """
    Assess anomaly-rate stability across k stratified folds.

    Because Isolation Forest is unsupervised, a conventional accuracy metric
    is unavailable. Instead, two proxy metrics are tracked per fold:
        - Silhouette score on the held-out fold's predictions.
        - Anomaly rate (fraction of fold labelled as outlier).

    Stable anomaly rates across folds indicate that the fitted contamination
    parameter is not over-sensitive to the particular training sample.

    Parameters
    ----------
    model_params : dict
        Valid IsolationForest parameters (from _valid_model_params).
    X_scaled : np.ndarray
        Standardised feature matrix (full dataset for this outcome).
    n_folds : int
        Number of cross-validation folds (default 5).

    Returns
    -------
    dict with keys:
        cv_silhouette_mean, cv_silhouette_std,
        cv_anomaly_rate_mean, cv_anomaly_rate_std,
        n_successful_folds
    """
    cv_scores, cv_anomaly_rates = [], []

    try:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=RANDOM_STATE)

        for train_idx, test_idx in kf.split(X_scaled, np.ones(len(X_scaled))):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]

            model = IsolationForest(**model_params, random_state=RANDOM_STATE)
            model.fit(X_train)
            test_labels = model.predict(X_test)

            if len(np.unique(test_labels)) > 1:
                try:
                    cv_scores.append(silhouette_score(X_test, test_labels))
                except Exception:
                    pass

            cv_anomaly_rates.append(np.mean(test_labels == -1))

    except Exception as exc:
        safe_print(f"    Cross-validation failed: {exc}")

    return {
        "cv_silhouette_mean":    np.mean(cv_scores) if cv_scores else 0,
        "cv_silhouette_std":     np.std(cv_scores) if cv_scores else 0,
        "cv_anomaly_rate_mean":  np.mean(cv_anomaly_rates) if cv_anomaly_rates else 0,
        "cv_anomaly_rate_std":   np.std(cv_anomaly_rates) if cv_anomaly_rates else 0,
        "n_successful_folds":    len(cv_scores),
    }


# =============================================================================
# 6. Main training pipeline
# =============================================================================

def run_anomaly_detection_pipeline(df):
    """
    Orchestrate outcome-stratified anomaly detection across all pregnancy outcomes.

    For each outcome stratum the function:
        1. Prepares the feature matrix.
        2. Tunes hyperparameters via grid search.
        3. Fits the final Isolation Forest on all stratum records.
        4. Runs 5-fold cross-validation.
        5. Stores anomaly scores, labels, and fitted objects.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'outcomef' column.

    Returns
    -------
    results : dict
        Keyed by outcome_type; each value is a dict with model artefacts
        and per-record anomaly labels / scores.
    """
    results = {}
    outcome_types = df["outcomef"].unique()
    safe_print(f"Outcome strata detected: {list(outcome_types)}")

    for outcome_type in outcome_types:
        safe_print(f"\n{'='*60}")
        safe_print(f"PROCESSING: {outcome_type}")
        safe_print(f"{'='*60}")

        try:
            X, features, outcome_data = prepare_outcome_specific_data(
                df, outcome_type
            )

            if len(X) < MIN_SAMPLES_PER_OUTCOME:
                safe_print(
                    f"  Skipping {outcome_type}: only {len(X)} samples "
                    f"(minimum {MIN_SAMPLES_PER_OUTCOME})."
                )
                continue

            best_params, scaler = tune_isolation_forest(X, outcome_type, features)
            X_scaled = scaler.transform(X)
            model_params = _valid_model_params(best_params)

            final_model = IsolationForest(**model_params,
                                          random_state=RANDOM_STATE)
            final_model.fit(X_scaled)
            anomaly_scores  = final_model.decision_function(X_scaled)
            anomaly_labels  = final_model.predict(X_scaled)

            safe_print(f"\n  Running {N_CV_FOLDS}-fold cross-validation ...")
            cv_results = cross_validate_model(model_params, X_scaled)

            results[outcome_type] = {
                "model":          final_model,
                "scaler":         scaler,
                "best_params":    best_params,
                "model_params":   model_params,
                "anomaly_scores": anomaly_scores,
                "anomaly_labels": anomaly_labels,
                "features_used":  features,
                "outcome_data":   outcome_data,
                "n_anomalies":    int(np.sum(anomaly_labels == -1)),
                "anomaly_rate":   float(np.mean(anomaly_labels == -1)),
                "n_samples":      len(X),
                "cv_results":     cv_results,
            }

            safe_print(
                f"  Anomalies detected : "
                f"{results[outcome_type]['n_anomalies']} "
                f"({results[outcome_type]['anomaly_rate']:.1%})"
            )
            safe_print(
                f"  CV silhouette      : "
                f"{cv_results['cv_silhouette_mean']:.3f} "
                f"+/- {cv_results['cv_silhouette_std']:.3f}"
            )
            safe_print(
                f"  Successful folds   : "
                f"{cv_results['n_successful_folds']}/{N_CV_FOLDS}"
            )

        except Exception as exc:
            safe_print(f"  Pipeline failed for {outcome_type}: {exc}")

    return results


# =============================================================================
# 7. Feature importance — permutation method
# =============================================================================

def permutation_importance_analysis(results, n_permutations=N_PERMUTATIONS):
    """
    Estimate feature importance by measuring anomaly-rate change under
    random column permutation.

    For each feature, the column is shuffled n_permutations times; the mean
    absolute change in the stratum-level anomaly rate is recorded as the
    importance score. A higher score indicates that disrupting that feature's
    signal meaningfully alters the model's anomaly classifications.

    This method is model-agnostic and appropriate for tree-ensemble models
    such as Isolation Forest, where internal feature weights are not
    straightforwardly interpretable.

    Parameters
    ----------
    results : dict
        Output from run_anomaly_detection_pipeline().
    n_permutations : int
        Number of shuffle replicates per feature (default 30).

    Returns
    -------
    permutation_results : dict
        Keyed by outcome_type; values are dicts keyed by feature name,
        each containing 'importance_mean', 'importance_std', 'description'.
    """
    safe_print(f"\n{'='*80}")
    safe_print("PERMUTATION IMPORTANCE ANALYSIS")
    safe_print(f"{'='*80}")

    permutation_results = {}

    for outcome, result in results.items():
        safe_print(f"\n  Processing {outcome} ...")

        try:
            model    = result["model"]
            scaler   = result["scaler"]
            features = result["features_used"]
            X_raw    = result["outcome_data"][features].copy()

            # Replicate training-time imputation
            X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
            for col in X_raw.columns:
                if X_raw[col].isna().any():
                    X_raw[col] = X_raw[col].fillna(X_raw[col].median())

            X_scaled = scaler.transform(X_raw.values)
            original_labels = model.predict(X_scaled)
            original_anomaly_rate = np.mean(original_labels == -1)

            feature_importance = {}

            for feat_idx, feat_name in enumerate(features):
                rate_changes = []

                for _ in range(n_permutations):
                    try:
                        X_perm = X_scaled.copy()
                        np.random.shuffle(X_perm[:, feat_idx])
                        perm_rate = np.mean(model.predict(X_perm) == -1)
                        rate_changes.append(
                            abs(original_anomaly_rate - perm_rate)
                        )
                    except Exception:
                        rate_changes.append(0.0)

                feature_importance[feat_name] = {
                    "importance_mean":     float(np.mean(rate_changes)),
                    "importance_std":      float(np.std(rate_changes)),
                    "description":         FEATURE_LABELS.get(feat_name, feat_name),
                    "n_permutations":      len(rate_changes),
                    "original_anomaly_rate": original_anomaly_rate,
                }

            permutation_results[outcome] = feature_importance
            safe_print(f"  Completed {len(feature_importance)} features.")

        except Exception as exc:
            safe_print(f"  Failed for {outcome}: {exc}")

    return permutation_results


# =============================================================================
# 8. Feature importance — effect size (Cohen's d)
# =============================================================================

def effect_size_analysis(results):
    """
    Quantify the separation between anomalous and normal records per feature
    using Cohen's d (pooled standard deviation).

    Cohen's d complements the permutation approach: permutation importance
    captures how much a feature drives the model's decisions, while Cohen's d
    describes how much anomalous records actually differ from normal records
    on each feature in raw (unscaled) units.

    Parameters
    ----------
    results : dict
        Output from run_anomaly_detection_pipeline().

    Returns
    -------
    effect_size_results : dict
        Keyed by outcome_type; values are dicts keyed by feature name,
        each containing 'effect_size', 'anomaly_mean', 'normal_mean',
        'difference', 'description'.
    """
    safe_print(f"\n{'='*80}")
    safe_print("EFFECT SIZE ANALYSIS (Cohen's d)")
    safe_print(f"{'='*80}")

    effect_size_results = {}

    for outcome, result in results.items():
        safe_print(f"\n  Analysing {outcome} ...")
        feature_effects = {}

        anomalies_mask = result["anomaly_labels"] == -1
        normal_mask    = result["anomaly_labels"] == 1
        X_values       = result["outcome_data"][result["features_used"]].values

        for feat_idx, feat_name in enumerate(result["features_used"]):
            try:
                anom_vals   = X_values[anomalies_mask, feat_idx]
                normal_vals = X_values[normal_mask, feat_idx]

                if len(anom_vals) > 0 and len(normal_vals) > 0:
                    pooled_std = np.sqrt(
                        (np.std(anom_vals) ** 2 + np.std(normal_vals) ** 2) / 2
                    )
                    if pooled_std > 0:
                        effect_size = (
                            np.mean(anom_vals) - np.mean(normal_vals)
                        ) / pooled_std

                        feature_effects[feat_name] = {
                            "effect_size":  abs(effect_size),
                            "anomaly_mean": float(np.mean(anom_vals)),
                            "normal_mean":  float(np.mean(normal_vals)),
                            "difference":   float(np.mean(anom_vals) - np.mean(normal_vals)),
                            "description":  FEATURE_LABELS.get(feat_name, feat_name),
                        }
            except Exception:
                continue

        effect_size_results[outcome] = feature_effects
        safe_print(f"  Analysed {len(feature_effects)} features.")

    return effect_size_results


# =============================================================================
# 9. Summary display helpers
# =============================================================================

def print_pipeline_summary(results):
    """Print a formatted summary table of per-outcome model performance."""
    safe_print(f"\n{'='*80}")
    safe_print("FINAL RESULTS SUMMARY")
    safe_print(f"{'='*80}")

    summary_rows = []
    for outcome, result in results.items():
        params = result["best_params"]
        cv     = result["cv_results"]
        summary_rows.append({
            "Outcome":       outcome,
            "Samples":       result["n_samples"],
            "Anomalies":     result["n_anomalies"],
            "Anomaly Rate":  f"{result['anomaly_rate']:.1%}",
            "Contamination": f"{params['contamination']:.3f}",
            "Estimators":    params["n_estimators"],
            "CV Silhouette": (
                f"{cv['cv_silhouette_mean']:.3f} "
                f"+/- {cv['cv_silhouette_std']:.3f}"
            ),
            "CV Folds":      f"{cv['n_successful_folds']}/{N_CV_FOLDS}",
        })
        safe_print(f"\n  {outcome}:")
        safe_print(f"     Samples    : {result['n_samples']:,}")
        safe_print(f"     Anomalies  : {result['n_anomalies']} "
                   f"({result['anomaly_rate']:.1%})")
        safe_print(f"     Contam.    : {params['contamination']:.3f}, "
                   f"n_est={params['n_estimators']}")
        safe_print(f"     CV Silh.   : "
                   f"{cv['cv_silhouette_mean']:.3f} "
                   f"+/- {cv['cv_silhouette_std']:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    safe_print(f"\n{'='*80}")
    safe_print("SUMMARY TABLE")
    safe_print(f"{'='*80}")
    safe_print(summary_df.to_string(index=False))
    return summary_df


def print_anomaly_characteristics(results):
    """
    Report mean +/- SD of key GA metrics for anomalous vs. normal records,
    per outcome stratum, to support clinical interpretation.
    """
    safe_print(f"\n{'='*80}")
    safe_print("ANOMALY CHARACTERISTICS BY OUTCOME")
    safe_print(f"{'='*80}")

    metrics = ["ga_del", "ga_edd", "ga_zscore", "ga_deviation"]

    for outcome, result in results.items():
        safe_print(
            f"\n  {outcome}: {result['n_anomalies']} anomalies "
            f"({result['anomaly_rate']:.1%})"
        )

        anom_mask   = result["anomaly_labels"] == -1
        normal_mask = result["anomaly_labels"] == 1

        for metric in metrics:
            if metric in result["outcome_data"].columns:
                try:
                    anom_v   = result["outcome_data"].loc[anom_mask, metric]
                    normal_v = result["outcome_data"].loc[normal_mask, metric]
                    if len(anom_v) > 0:
                        safe_print(f"    {metric}:")
                        safe_print(
                            f"       Anomalies : "
                            f"{anom_v.mean():.2f} +/- {anom_v.std():.2f}"
                        )
                        safe_print(
                            f"       Normal    : "
                            f"{normal_v.mean():.2f} +/- {normal_v.std():.2f}"
                        )
                except Exception:
                    pass


# =============================================================================
# 10. Output export
# =============================================================================

def export_anomaly_datasets(results, original_df):
    """
    Merge per-outcome anomaly labels back into the full dataset and export
    three CSV files plus a summary table.

    The clean dataset (anomalies removed) is the primary input to Stage 2
    (02_lmm_ga_estimation.R).

    Parameters
    ----------
    results : dict
        Output from run_anomaly_detection_pipeline().
    original_df : pd.DataFrame
        Full original dataset (all columns preserved).

    Returns
    -------
    dict with keys:
        combined_data   — all records + anomaly columns
        clean_data      — non-anomalous records
        anomaly_data    — anomalous records sorted by score
        summary         — pd.DataFrame summary table
    """
    safe_print(f"\n{'='*80}")
    safe_print("EXPORTING DATASETS")
    safe_print(f"{'='*80}")

    combined = original_df.copy()
    combined["is_anomaly"]                 = False
    combined["anomaly_score"]              = np.nan
    combined["anomaly_outcome"]            = ""
    combined["anomaly_rank_within_outcome"] = np.nan

    total_anomalies = 0
    outcome_counts  = {}

    for outcome, result in results.items():
        safe_print(f"\n  Merging flags for {outcome} ...")
        outcome_idx = combined[combined["outcomef"] == outcome].index

        if len(outcome_idx) != len(result["anomaly_labels"]):
            safe_print(
                f"    Index mismatch: {len(outcome_idx)} rows in original "
                f"vs {len(result['anomaly_labels'])} in results. Skipping."
            )
            continue

        anom_mask = result["anomaly_labels"] == -1
        n_anom    = int(anom_mask.sum())

        combined.loc[outcome_idx, "is_anomaly"]    = anom_mask
        combined.loc[outcome_idx, "anomaly_score"] = result["anomaly_scores"]
        combined.loc[outcome_idx, "anomaly_outcome"] = outcome

        # Rank within outcome: rank 1 = most anomalous
        ranks = np.argsort(np.argsort(-result["anomaly_scores"])) + 1
        combined.loc[outcome_idx, "anomaly_rank_within_outcome"] = ranks

        outcome_counts[outcome] = n_anom
        total_anomalies += n_anom
        safe_print(f"    Flagged {n_anom} anomalies.")

    clean_data = combined[~combined["is_anomaly"]].copy()
    anomaly_data = (
        combined[combined["is_anomaly"]]
        .copy()
        .sort_values("anomaly_score", ascending=False)
    )

    # --- CSV exports ---
    combined.to_csv("all_data_with_anomaly_flags.csv", index=False)
    clean_data.to_csv("clean_data_for_lmm.csv", index=False)
    anomaly_data.to_csv("anomalies_for_review.csv", index=False)

    summary_rows = []
    for outcome, result in results.items():
        n_tot = len(combined[combined["outcomef"] == outcome])
        n_an  = outcome_counts.get(outcome, 0)
        summary_rows.append({
            "outcome":            outcome,
            "total_records":      n_tot,
            "anomalies_detected": n_an,
            "anomaly_rate":       n_an / n_tot if n_tot > 0 else np.nan,
            "contamination_param": result["best_params"]["contamination"],
            "cv_silhouette":      result["cv_results"]["cv_silhouette_mean"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("anomaly_detection_summary.csv", index=False)

    safe_print(f"\n  Total records   : {len(combined):,}")
    safe_print(
        f"  Total anomalies : {total_anomalies:,} "
        f"({total_anomalies/len(combined):.1%})"
    )
    safe_print(f"  Clean records   : {len(clean_data):,}")
    safe_print("\n  Files saved:")
    safe_print("    all_data_with_anomaly_flags.csv")
    safe_print("    clean_data_for_lmm.csv        <- input to Stage 2")
    safe_print("    anomalies_for_review.csv")
    safe_print("    anomaly_detection_summary.csv")

    return {
        "combined_data": combined,
        "clean_data":    clean_data,
        "anomaly_data":  anomaly_data,
        "summary":       summary_df,
    }


# =============================================================================
# 11. Manuscript tables
# =============================================================================

def generate_manuscript_tables(results, effect_size_results):
    """
    Generate two publication-ready CSV tables:
        table1_model_performance.csv  — per-outcome model metrics
        table2_feature_importance.csv — top-3 features by Cohen's d per outcome

    Parameters
    ----------
    results : dict
        Output from run_anomaly_detection_pipeline().
    effect_size_results : dict
        Output from effect_size_analysis().
    """
    safe_print(f"\n{'='*80}")
    safe_print("GENERATING MANUSCRIPT TABLES")
    safe_print(f"{'='*80}")

    # Table 1: model performance
    perf_rows = []
    for outcome, result in results.items():
        params = result["best_params"]
        cv     = result["cv_results"]
        perf_rows.append({
            "Outcome":          outcome.replace("_", " ").title(),
            "n":                f"{result['n_samples']:,}",
            "Anomalies":        result["n_anomalies"],
            "Anomaly Rate":     f"{result['anomaly_rate']:.1%}",
            "Contamination":    f"{params['contamination']:.3f}",
            "Estimators":       params["n_estimators"],
            "CV Silhouette":    (
                f"{cv['cv_silhouette_mean']:.3f} "
                f"+/- {cv['cv_silhouette_std']:.3f}"
            ),
            "Tuning Silhouette": f"{params.get('silhouette_score', 0):.3f}",
        })

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv("table1_model_performance.csv", index=False)
    safe_print("\n  Saved: table1_model_performance.csv")
    safe_print(perf_df.to_string(index=False))

    # Table 2: top-3 features by Cohen's d per outcome
    imp_rows = []
    for outcome, features in effect_size_results.items():
        if features:
            top3 = sorted(
                features.items(), key=lambda x: x[1]["effect_size"], reverse=True
            )[:3]
            for rank, (feat_name, stats) in enumerate(top3, 1):
                imp_rows.append({
                    "Outcome":      outcome.replace("_", " ").title(),
                    "Rank":         rank,
                    "Feature":      stats["description"],
                    "Effect Size":  f"{stats['effect_size']:.3f}",
                    "Anomaly Mean": f"{stats['anomaly_mean']:.2f}",
                    "Normal Mean":  f"{stats['normal_mean']:.2f}",
                    "Difference":   f"{stats['difference']:+.2f}",
                })

    imp_df = pd.DataFrame(imp_rows)
    imp_df.to_csv("table2_feature_importance.csv", index=False)
    safe_print("\n  Saved: table2_feature_importance.csv")
    safe_print(imp_df.to_string(index=False))

    return perf_df, imp_df

# =============================================================================
# 12. Entry point
# =============================================================================

if __name__ == "__main__":

    safe_print("=" * 80)
    safe_print("MiMBa GA Estimation Pipeline — Stage 1: Anomaly Detection")
    safe_print("=" * 80)

    # --- Load data ---
    safe_print(f"\nLoading data from: {INPUT_FILE}")
    dsk = pd.read_csv(INPUT_FILE)
    safe_print(f"  Rows    : {len(dsk):,}")
    safe_print(f"  Columns : {list(dsk.columns)}")

    # --- Stage 1a: Isolation Forest ---
    final_results = run_anomaly_detection_pipeline(dsk)

    # --- Stage 1b: Summarise ---
    summary_df = print_pipeline_summary(final_results)
    print_anomaly_characteristics(final_results)

    # --- Stage 1c: Feature importance ---
    permutation_results = permutation_importance_analysis(
        final_results, n_permutations=N_PERMUTATIONS
    )
    effect_size_results = effect_size_analysis(final_results)

    # --- Stage 1d: Manuscript tables ---
    generate_manuscript_tables(final_results, effect_size_results)
    generate_latex_feature_table(permutation_results, effect_size_results)

    # --- Stage 1e: Export datasets ---
    export_results = export_anomaly_datasets(final_results, dsk)

    # --- Done ---
    safe_print(f"\n{'='*80}")
    safe_print(
        f"STAGE 1 COMPLETE — {len(final_results)} outcome strata processed."
    )
    safe_print(
        "  Next: run 02_lmm_ga_estimation.R "
        "using clean_data_for_lmm.csv as input."
    )
    safe_print("=" * 80)
