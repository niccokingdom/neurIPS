# -----------------------------------------------------------
# extended dual-model utilities
# -----------------------------------------------------------
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from typing import Dict, List, Tuple, Sequence

# ---------------- baseline trainer (unchanged) -------------
def _train_one_lgb(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   cat_cols: Sequence[str] | None = None,
                   sample_weight: pd.Series | None = None,
                   params: Dict | None = None) -> lgb.LGBMRegressor:

    if params is None:
        params = dict(
            n_estimators=1000, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
        )

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        eval_metric="l1",
        categorical_feature=cat_cols if cat_cols else "auto",
        sample_weight=sample_weight

    )
    return model

def plot_fi(model: lgb.LGBMRegressor,
            cols: List[str],
            ax: plt.Axes,
            top_n: int = 20,
            title: str = "") -> None:
    """Plots feature importances for a given model."""
    fi = pd.DataFrame({"feature": cols, "gain": model.feature_importances_})
    fi = fi[fi["feature"] != "id"]          # ← drop 'id'
    fi = fi.sort_values("gain", ascending=False).head(top_n)[::-1]
    ax.barh(fi["feature"], fi["gain"])
    ax.set_title(title)
    ax.set_xlabel("gain")

# ---------------- helper: build feature set -------------
def _get_feature_cols(df: pd.DataFrame, target_cols: Sequence[str]) -> List[str]:
    """Gets all feature columns, excluding targets and identifiers."""
    exclude = set(target_cols)
    exclude.update(c for c in df.columns if c.startswith(("id", "SMILES", "can_smiles")))
    exclude.update(c for c in df.columns for t in target_cols if c.startswith(f"{t}_"))
    exclude.update(c for c in df.columns for t in target_cols if c.startswith(f"weight_{t}"))
    return [c for c in df.columns if c not in exclude]

# ---------------- main single-model trainer -----------------
def train_single_model_and_report(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_cols: Tuple[str, ...] = ("Tg", "FFV", "Tc", "Density", "Rg"),
        cluster_cols=("cb_k20", "cb_hdb25", "mf_k20", "mf_hdb25"),
        random_state=42,
        lgb_params: Dict | None = None
) -> Tuple[Dict[str, float], Dict[str, lgb.LGBMRegressor]]:
    """
    Trains one LGBM per target using all available features.
    Returns metrics and trained models.
    """
    feature_cols = _get_feature_cols(df_train, target_cols)
    cat_cols = [c for c in cluster_cols if c in feature_cols]

    metrics: Dict[str, float] = {}
    models: Dict[str, lgb.LGBMRegressor] = {}

    for tgt in target_cols:
        sub_train = df_train.dropna(subset=[tgt])
        sub_test = df_test.dropna(subset=[tgt])
        sub_weight = sub_train[f'weight_{tgt}'] if f'weight_{tgt}' in sub_train else None

        y_tr = sub_train[tgt]
        X_tr = sub_train[feature_cols]

        y_te = sub_test[tgt]
        X_te = sub_test[feature_cols]

        model = _train_one_lgb(X_tr, y_tr,
                               cat_cols=cat_cols,
                               params=lgb_params,
                               sample_weight=sub_weight
                            )

        p = model.predict(X_te)
        mae = mean_absolute_error(y_te, p)
        metrics[tgt] = mae
        models[tgt] = model

    return metrics, models, feature_cols