"""XGBoost Risk Prediction training pipeline.

Trains a supervised XGBoost classifier on the full feature-engineered dataset
to predict the binary ``Accident`` target.  This is the primary model for the
Risk Prediction module.

Usage:
    python -m training.train_risk_prediction

    # Or programmatically:
    from training.train_risk_prediction import run_pipeline
    results = run_pipeline()

Outputs:
    models/risk_prediction_xgboost.pkl     — trained XGBClassifier
    models/risk_prediction_scaler.pkl      — fitted StandardScaler
    models/risk_prediction_features.json   — feature importance + column order
    models/risk_prediction_report.json     — evaluation metrics report

Training remains **outside** the runtime application.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except (ImportError, OSError):
    # Fallback when XGBoost is unavailable (e.g. missing libomp on macOS)
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier  # type: ignore[assignment]

    _HAS_XGBOOST = False

from training.feature_engineering import get_risk_prediction_features

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_FEATURES_PATH = "datasets/processed/features.csv"
DEFAULT_MODEL_PATH = "models/risk_prediction_xgboost.pkl"
DEFAULT_SCALER_PATH = "models/risk_prediction_scaler.pkl"
DEFAULT_FEATURES_META_PATH = "models/risk_prediction_features.json"
DEFAULT_REPORT_PATH = "models/risk_prediction_report.json"

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation
CV_FOLDS = 5

# XGBoost default hyperparameters (tuned via grid search)
DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Params that only XGBoost understands (removed when falling back to sklearn)
_XGBOOST_ONLY_PARAMS = {
    "min_child_weight", "colsample_bytree", "gamma", "reg_alpha",
    "reg_lambda", "objective", "eval_metric", "use_label_encoder",
    "n_jobs", "scale_pos_weight",
}


def _filter_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove XGBoost-only params when running on sklearn fallback."""
    if _HAS_XGBOOST:
        return params
    return {k: v for k, v in params.items() if k not in _XGBOOST_ONLY_PARAMS}

# Hyperparameter search grid (used by tune_hyperparameters)
PARAM_GRID: Dict[str, list] = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [200, 300, 500],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result containers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class EvaluationResult:
    """Holds evaluation metrics for the risk prediction model."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: list  # Serialisable (list of lists)
    report: str
    total_samples: int
    test_samples: int
    train_samples: int
    positive_count: int
    feature_count: int
    cv_scores: list = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    best_params: dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        d = asdict(self)
        d["confusion"] = self.confusion
        return d


@dataclass
class FeatureImportance:
    """Feature importance + column order metadata."""

    feature_names: list
    importances: list
    column_order: list

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def top_n(self) -> list[tuple[str, float]]:
        """Top 10 features by importance."""
        pairs = sorted(
            zip(self.feature_names, self.importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return pairs[:10]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Load and prepare data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_features(filepath: str | Path = DEFAULT_FEATURES_PATH) -> pd.DataFrame:
    """Load the feature-engineered dataset."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Features file not found: {filepath}")
    df = pd.read_csv(filepath)
    logger.info("Loaded features: %d rows × %d cols", len(df), len(df.columns))
    return df


def prepare_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    """Prepare train/test split with scaling.

    Args:
        df: Full feature DataFrame.
        test_size: Fraction for test set.
        random_state: Random seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test, feature_columns, scaler)
    """
    feature_cols = get_risk_prediction_features(df)
    logger.info("Selected %d features for Risk Prediction", len(feature_cols))

    X = df[feature_cols].values
    y = df["Accident"].values

    # Handle NaN
    if np.isnan(X).any():
        nan_count = int(np.isnan(X).sum())
        logger.warning("Found %d NaN values in features — filling with 0", nan_count)
        X = np.nan_to_num(X, nan=0.0)

    # Stratified split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(
        "Data split: train=%d (%.1f%% positive), test=%d (%.1f%% positive)",
        len(X_train), y_train.mean() * 100,
        len(X_test), y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test, feature_cols, scaler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Train model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> XGBClassifier:
    """Train an XGBoost classifier.

    Automatically computes ``scale_pos_weight`` to handle class imbalance.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.
        params: XGBoost hyperparameters.  Uses DEFAULT_PARAMS if None.

    Returns:
        Trained XGBClassifier.
    """
    params = dict(params or DEFAULT_PARAMS)

    # Handle class imbalance
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    if pos_count > 0:
        params.setdefault("scale_pos_weight", neg_count / pos_count)
        logger.info(
            "Class balance: neg=%d, pos=%d, scale_pos_weight=%.2f",
            neg_count, pos_count, params["scale_pos_weight"],
        )

    filtered = _filter_params(params)
    model = XGBClassifier(**filtered)
    if _HAS_XGBOOST:
        model.fit(X_train, y_train, verbose=False)
    else:
        model.fit(X_train, y_train)

    logger.info(
        "XGBoost trained: n_estimators=%d, max_depth=%d, lr=%.3f",
        params.get("n_estimators", 0),
        params.get("max_depth", 0),
        params.get("learning_rate", 0),
    )
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Cross-validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def cross_validate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    cv_folds: int = CV_FOLDS,
) -> tuple[list[float], float, float]:
    """Run stratified k-fold cross-validation.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.
        params: XGBoost params.
        cv_folds: Number of folds.

    Returns:
        (scores, mean, std) — per-fold ROC-AUC scores, mean, and std.
    """
    params = dict(params or DEFAULT_PARAMS)

    # Handle class imbalance
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    if pos > 0:
        params.setdefault("scale_pos_weight", neg / pos)

    model = XGBClassifier(**_filter_params(params))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    logger.info(
        "Cross-validation (%d-fold): ROC-AUC = %.4f ± %.4f",
        cv_folds, mean_score, std_score,
    )
    return scores.tolist(), mean_score, std_score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Hyperparameter tuning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[Dict[str, list]] = None,
    cv_folds: int = 3,
    n_iter: int = 20,
) -> Dict[str, Any]:
    """Randomised search over XGBoost hyperparameters.

    Uses RandomizedSearchCV for efficiency on large search spaces.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.
        param_grid: Parameter distributions to sample from.
        cv_folds: CV folds for tuning.
        n_iter: Number of random combinations to try.

    Returns:
        Best parameters dict.
    """
    from sklearn.model_selection import RandomizedSearchCV

    param_grid = param_grid or PARAM_GRID

    base_params = dict(DEFAULT_PARAMS)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    if pos > 0:
        base_params["scale_pos_weight"] = neg / pos

    model = XGBClassifier(**base_params)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best = search.best_params_
    logger.info("Best params (ROC-AUC=%.4f): %s", search.best_score_, best)
    return best


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Evaluate model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
    cv_scores: Optional[list[float]] = None,
    best_params: Optional[dict] = None,
) -> tuple[EvaluationResult, FeatureImportance]:
    """Evaluate the trained model on the held-out test set.

    Args:
        model: Trained XGBClassifier.
        X_test: Scaled test features.
        y_test: Test labels.
        X_train: Training features (for size reporting).
        y_train: Training labels (for size reporting).
        feature_cols: Feature column names (for importance).
        cv_scores: Cross-validation scores (if computed).
        best_params: Best hyperparameters (if tuned).

    Returns:
        (EvaluationResult, FeatureImportance)
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=["Normal", "Accident"],
    )

    cv_scores = cv_scores or []
    cv_mean = float(np.mean(cv_scores)) if cv_scores else 0.0
    cv_std = float(np.std(cv_scores)) if cv_scores else 0.0

    result = EvaluationResult(
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        roc_auc=round(roc, 4),
        confusion=cm,
        report=report,
        total_samples=len(y_train) + len(y_test),
        test_samples=len(y_test),
        train_samples=len(y_train),
        positive_count=int(y_test.sum()),
        feature_count=len(feature_cols),
        cv_scores=[round(s, 4) for s in cv_scores],
        cv_mean=round(cv_mean, 4),
        cv_std=round(cv_std, 4),
        best_params=best_params or {},
    )

    # Feature importance
    importances = model.feature_importances_.tolist()
    feat_importance = FeatureImportance(
        feature_names=feature_cols,
        importances=importances,
        column_order=feature_cols,
    )

    logger.info(
        "Evaluation: Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  AUC=%.4f",
        result.accuracy, result.precision, result.recall, result.f1, result.roc_auc,
    )
    return result, feat_importance


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Save artefacts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def save_model(
    model: XGBClassifier,
    scaler: StandardScaler,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
) -> tuple[Path, Path]:
    """Persist the trained model and scaler."""
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    for p in (model_path, scaler_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model → %s", model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler → %s", scaler_path)

    return model_path, scaler_path


def save_feature_importance(
    feat_importance: FeatureImportance,
    filepath: str | Path = DEFAULT_FEATURES_META_PATH,
) -> Path:
    """Save feature importance and column order as JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(feat_importance.to_dict(), f, indent=2)
    logger.info("Saved feature metadata → %s", filepath)
    return filepath


def save_evaluation_report(
    result: EvaluationResult,
    filepath: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Save evaluation metrics as JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Saved evaluation report → %s", filepath)
    return filepath


def load_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
) -> tuple[XGBClassifier, StandardScaler]:
    """Load a previously trained model and scaler from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Loaded model from %s and scaler from %s", model_path, scaler_path)
    return model, scaler


def load_feature_metadata(
    filepath: str | Path = DEFAULT_FEATURES_META_PATH,
) -> FeatureImportance:
    """Load feature importance and column order from JSON."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return FeatureImportance(**data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_pipeline(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    features_meta_path: str | Path = DEFAULT_FEATURES_META_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    *,
    do_tuning: bool = False,
    tuning_n_iter: int = 20,
) -> EvaluationResult:
    """Execute the full XGBoost Risk Prediction training pipeline.

    Steps:
        1. Load features
        2. Prepare data (stratified train/test split + scaling)
        3. (Optional) Hyperparameter tuning
        4. Train XGBoost model
        5. Cross-validation
        6. Evaluate on test set
        7. Save all artefacts

    Args:
        features_path: Path to feature CSV.
        model_path: Output path for model pickle.
        scaler_path: Output path for scaler pickle.
        features_meta_path: Output path for feature importance JSON.
        report_path: Output path for evaluation report JSON.
        do_tuning: Whether to run hyperparameter tuning.
        tuning_n_iter: Number of random search iterations.

    Returns:
        EvaluationResult with full metrics.
    """
    # 1. Load
    df = load_features(features_path)

    # 2. Prepare
    X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_data(df)

    # 3. Hyperparameter tuning (optional)
    best_params = None
    if do_tuning:
        logger.info("Starting hyperparameter tuning (%d iterations)...", tuning_n_iter)
        best_params = tune_hyperparameters(
            X_train, y_train, n_iter=tuning_n_iter,
        )
        # Merge best params with defaults
        params = {**DEFAULT_PARAMS, **best_params}
    else:
        params = DEFAULT_PARAMS

    # 4. Train
    model = train_model(X_train, y_train, params=params)

    # 5. Cross-validation
    cv_scores, cv_mean, cv_std = cross_validate(X_train, y_train, params=params)

    # 6. Evaluate
    result, feat_importance = evaluate_model(
        model, X_test, y_test, X_train, y_train,
        feature_cols, cv_scores=cv_scores, best_params=best_params,
    )

    # 7. Save
    save_model(model, scaler, model_path, scaler_path)
    save_feature_importance(feat_importance, features_meta_path)
    save_evaluation_report(result, report_path)

    # Print report
    _print_report(result, feat_importance, feature_cols, model_path, scaler_path)

    return result


def _print_report(
    result: EvaluationResult,
    feat_importance: FeatureImportance,
    feature_cols: list[str],
    model_path: str | Path,
    scaler_path: str | Path,
) -> None:
    """Print a human-readable training report to stdout."""
    print("\n" + "=" * 64)
    print("  XGBOOST RISK PREDICTION — TRAINING REPORT")
    print("=" * 64)
    print(f"  Training samples:   {result.train_samples:,}")
    print(f"  Test samples:       {result.test_samples:,}")
    print(f"  Features used:      {result.feature_count}")
    print(f"  Positive (test):    {result.positive_count:,}")
    print("-" * 64)
    print(f"  Accuracy:           {result.accuracy:.4f}")
    print(f"  Precision:          {result.precision:.4f}")
    print(f"  Recall:             {result.recall:.4f}")
    print(f"  F1 Score:           {result.f1:.4f}")
    print(f"  ROC-AUC:            {result.roc_auc:.4f}")
    print("-" * 64)
    if result.cv_scores:
        print(f"  CV ROC-AUC:         {result.cv_mean:.4f} ± {result.cv_std:.4f}")
        print(f"  CV Folds:           {len(result.cv_scores)}")
    print("-" * 64)
    print("  Confusion Matrix:")
    print(f"    TN={result.confusion[0][0]:,}  FP={result.confusion[0][1]:,}")
    print(f"    FN={result.confusion[1][0]:,}  TP={result.confusion[1][1]:,}")
    print("-" * 64)
    print(result.report)
    print("-" * 64)
    print("  Top 10 Feature Importances:")
    for name, imp in feat_importance.top_n:
        print(f"    {name:40s} {imp:.4f}")
    print("-" * 64)
    print(f"  Model saved:   {model_path}")
    print(f"  Scaler saved:  {scaler_path}")
    print("=" * 64)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Train XGBoost Risk Prediction model")
    parser.add_argument(
        "--tune", action="store_true",
        help="Run hyperparameter tuning (slower)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=20,
        help="Number of tuning iterations (default: 20)",
    )
    args = parser.parse_args()

    run_pipeline(do_tuning=args.tune, tuning_n_iter=args.n_iter)
