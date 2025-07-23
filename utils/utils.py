import pandas as pd
import numpy as np

# ------------ 1.  Merge embeddings correctly ----------------
def build_feature_matrix(
        augmented_path: str,
        embeds_meta_path: str,
        embeds_npz_path: str
) -> pd.DataFrame:
    """
    Returns a DataFrame that has:
        • all columns from the augmented table
        • 1 152 numeric columns  cb_0 … cb_383  mf_0 … mf_767
    The embeddings are aligned by 'can_smiles', so order is guaranteed.
    """
    df_aug = pd.read_parquet(augmented_path)

    # meta has at least the key 'can_smiles' in the same order as the npz arrays
    meta = pd.read_parquet(embeds_meta_path)[["can_smiles"]]
    vecs = np.load(embeds_npz_path)
    cb   = vecs["chemberta"].astype("float32")   # (N, 384)
    mf   = vecs["molformer"].astype("float32")    # (N, 768)

    # build an embeddings table with only the key + vectors
    emb_df = meta.copy()
    emb_df[[f"cb_{i}" for i in range(cb.shape[1])]] = cb
    emb_df[[f"mf_{i}" for i in range(mf.shape[1])]] = mf

    # safe one-to-one merge – will raise if duplicates exist
    df_full = df_aug.merge(
        emb_df,
        on="can_smiles",
        how="left",
        validate="one_to_one"
    )
    return df_full