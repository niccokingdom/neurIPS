import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from typing import List, Sequence
from pathlib import Path

# ---------- 1. Canonical SMILES ---------- #
# 1. Canonicalise a single SMILES -------------------------------------------- #
def canonical_smiles(s: str) -> str | None:
    mol = Chem.MolFromSmiles(s)
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True)

# 2. Apply to a whole DataFrame, drop failures -------------------------------- #
def add_canonical_column(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    tqdm.pandas(desc="↻ Canonicalising")
    df["can_smiles"] = df[smiles_col].progress_apply(canonical_smiles)
    return df.dropna(subset=["can_smiles"]).reset_index(drop=True)


# 3. Average duplicates INSIDE one file -------------------------------------- #
def collapse_duplicates(
    df: pd.DataFrame, targets: Sequence[str], id_col: str = "can_smiles"
) -> pd.DataFrame:
    return df.groupby(id_col, as_index=False)[list(targets)].mean()


# 4. Merge/append ONE extra file into train ---------------------------------- #
def merge_extra(
    train: pd.DataFrame,
    extra: pd.DataFrame,
    target: str,
    id_col: str = "can_smiles",
) -> pd.DataFrame:
    """Fill NaNs for overlapping molecules, then append brand-new ones."""
    # Fill missing in place
    map_vals = extra.set_index(id_col)[target]
    mask = train[target].isna() & train[id_col].isin(map_vals.index)
    train.loc[mask, target] = train.loc[mask, id_col].map(map_vals)

    # Append unseen molecules
    new_rows = extra[~extra[id_col].isin(train[id_col])]
    if not new_rows.empty:
        train = pd.concat([train, new_rows[[id_col, target]]], ignore_index=True)

    return train


# 5. End-to-end orchestrator -------------------------------------------------- #
def build_augmented_train(
    train_path: str | Path,
    extra_paths: List[str | Path],
    targets: Sequence[str] = ("Tg", "FFV", "Tc", "Density", "Rg"),
    smiles_col: str = "SMILES",
) -> pd.DataFrame:
    train = pd.read_csv(train_path)
    train = add_canonical_column(train, smiles_col)

    # make sure all 5 target columns exist
    for t in targets:
        if t not in train.columns:
            train[t] = np.nan

    for p in extra_paths:
        extra = pd.read_csv(p)
        extra = add_canonical_column(extra, smiles_col)

        present_ts = [t for t in targets if t in extra.columns]
        if not present_ts:
            continue

        extra = collapse_duplicates(extra, present_ts)

        for t in present_ts:
            train = merge_extra(train, extra[[ "can_smiles", t]], t)

    return train.reset_index(drop=True)