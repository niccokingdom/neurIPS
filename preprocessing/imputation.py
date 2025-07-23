# preprocessing/imputation.py  (overwrite the previous version)
# -------------------------------------------------------------
import miceforest as mf
import pandas as pd
from typing import Tuple, Optional, Dict, Any, Sequence
import numpy as np
import matplotlib.pyplot as plt

USELESS_COLS  = {"SMILES", "can_smiles", "id"}
CAT_COLS      = {"cb_k20", "cb_hdb25", "mf_k20", "mf_hdb25"}

def _prepare_for_mice(df: pd.DataFrame, targets: Sequence[str] | None = None, cat_cols: Sequence[str] = CAT_COLS) -> pd.DataFrame:
    """Clean frame so miceforest won't choke on invalid columns."""
    if targets is not None:
        df2 = df[targets].copy()
    else:
        df2 = df.copy()

    #  drop useless string columns
    df2 = df2.drop(columns=[col for col in USELESS_COLS if col in df2])

    # cast cluster IDs to category
    for c in cat_cols:
        if c in df2:
            df2[c] = df2[c].astype("category")

    obj_cols = df2.select_dtypes(include="object").columns.tolist()
    assert not obj_cols, f"convert object dtypes first: {obj_cols}"
    return df2


def impute_with_miceforest(
    df: pd.DataFrame,
    targets: Sequence[str] | None = None,
    n_datasets: int = 1,
    n_iterations: int = 3,
    random_state: int = 42,
    mean_match_cand: int = 0,
    lgb_params: Optional[Dict[str, Any]] = None,
    plot_diagnostics: bool = True,
    drop_cols: Sequence[str] = USELESS_COLS,   # allow user override
    cat_cols:  Sequence[str] = CAT_COLS,
) -> Tuple[pd.DataFrame, mf.ImputationKernel]:

    df_ready = _prepare_for_mice(df, targets, cat_cols)

    if lgb_params is None:
        lgb_params = dict(
            boosting="gbdt",
            #num_leaves=32,
            learning_rate=0.05,
            n_estimators=1000,
            #bagging_fraction=0.8,
            #feature_fraction_bynode=0.8,
            #categorical_feature=[f"name:{col}" for col in cat_cols if col in df_ready.columns],
            # "device": "gpu",
        )

    kernel = mf.ImputationKernel(
        data=df_ready,
        num_datasets=n_datasets,
        random_state=random_state,
        save_all_iterations_data=True
    )

    kernel.mice(
        n_iterations,
        mean_match_candidates=mean_match_cand,
        **lgb_params
    )

    if plot_diagnostics:
        kernel.plot_mean_convergence()
        kernel.plot_feature_importance(dataset=0)
    plt.show()
    completed = kernel.complete_data(dataset=0)

    # If targets were specified, return the original df with imputed target columns
    if targets is not None:
        result = df.copy()
        # Only replace the target columns that were imputed
        for col in completed.columns:
            if col in targets:
                result[col] = completed[col]
        return result, kernel
    else:
        return completed, kernel

def apply_imputation(
    df: pd.DataFrame,
    kernel: mf.ImputationKernel,
    dataset_idx: int = 0
) -> pd.DataFrame:
    """
    Apply a trained imputation kernel to new data.
    
    Args:
        df: DataFrame with missing values to impute
        kernel: Trained ImputationKernel object
        dataset_idx: Which imputed dataset to use (default: 0)
        
    Returns:
        DataFrame with imputed values
    """
    df_ready = _prepare_for_mice(df)
    imputed_data = kernel.impute_new_data(new_data=df_ready)
    return imputed_data.complete_data(dataset=dataset_idx)