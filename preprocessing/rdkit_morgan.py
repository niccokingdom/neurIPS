# ---------------------------------------------------------------------
# 1. Extra features for polymers
# ---------------------------------------------------------------------
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import rdFingerprintGenerator as rfg, Descriptors
import numpy as np
import pandas as pd
from tqdm.auto import tqdm



# ---------- a) RDKit descriptors (lightweight but useful) ------------
DESC_LIST = [
    "MolWt", "ExactMolWt", "FractionCSP3", "TPSA",
    "RingCount", "NumRotatableBonds",
    "NumHAcceptors", "NumHDonors", "MolLogP",
]

def add_all_rdkit_descriptors(df: pd.DataFrame,
                              smiles_col: str = "can_smiles",
                              prefix: str = "desc_") -> pd.DataFrame:
    """
    Compute *all* numeric 2-D RDKit descriptors for each SMILES in `df[smiles_col]`
    and return the original df with the new columns appended.

    Any descriptor that errors (rare) is set to NaN.
    """
    # 1 / list of all descriptor names RDKit currently knows
    desc_names = [name for name, _ in Descriptors._descList]      # ~208 items

    # 2 / a fast descriptor calculator object
    calc = MolecularDescriptorCalculator(desc_names)

    # 3 / helper to turn a SMILES into a list of descriptor values
    def _calc_one(smi: str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [np.nan] * len(desc_names)
        try:
            return list(calc.CalcDescriptors(mol))
        except Exception:
            # handle the very occasional divide-by-zero or valence error
            return [np.nan] * len(desc_names)

    tqdm.pandas(desc="RDKit descriptors (all)")
    desc_matrix = df[smiles_col].progress_apply(_calc_one).tolist()

    desc_df = pd.DataFrame(
        desc_matrix,
        columns=[f"{prefix}{n}" for n in desc_names],
        index=df.index,
    )
    return pd.concat([df, desc_df], axis=1)


# ---------- b) 256-int folded Morgan fingerprints --------------------
def add_morgan_fingerprints(df: pd.DataFrame,
                            smiles_col: str = "can_smiles",
                            radius: int = 2,
                            n_bits: int = 2048,
                            fold_to: int = 256,
                            prefix: str = "morgan_") -> pd.DataFrame:
    """
    Computes radius-`radius` Morgan bits, then folds 2048 bits → `fold_to`
    integers (little-endian, 8 bits per int).  Produces `fold_to` columns
    named  morgan_0 … morgan_255.
    """
    gen = rfg.GetMorganGenerator(radius=radius,
                                 fpSize=n_bits,
                                 includeChirality=True)

    def _fp_to_ints(smi: str):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return np.zeros(fold_to, dtype=np.uint8)
        bits = np.frombuffer(
            gen.GetFingerprint(m).ToBitString().encode("ascii"),
            dtype=np.uint8
        ) - 48                                    # '0'/'1' → 0/1
        ints = bits.reshape(fold_to, -1).dot(1 << np.arange(8))
        return ints.astype(np.uint8)

    tqdm.pandas(desc="Morgan fold-256")
    int_vectors = df[smiles_col].progress_apply(_fp_to_ints).tolist()
    morgan_df = pd.DataFrame(
        np.vstack(int_vectors),
        columns=[f"{prefix}{i}" for i in range(fold_to)],
        index=df.index
    )
    return pd.concat([df, morgan_df], axis=1)
