#!/usr/bin/env python3
# ----------------------------------------------------------------------
#  embed_polymer.py
#
#  Compute ChemBERTa-77 M (384-d) and MolFormer-XL (768-d) embeddings
#  for a list of canonical SMILES strings.  Designed for Kaggle /
#  offline environments: if the models are not on disk they are
#  downloaded once to the local HuggingFace cache (~500 MB total).
#
#  Usage (CLI):
#     python embed_polymer.py \
#         --input  data/train_augmented.csv \
#         --output data/train_embeds   \
#         --batch-size 32              \
#         --device cuda                \
#         --save-format parquet
#
#  Usage (from a notebook):
#     from embed_polymer import generate_embeddings
#     df = pd.read_csv("train_augmented.csv")
#     df_emb = generate_embeddings(df, device="cuda")
#
#  ---------------------------------------------------------------------

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# ----------------------------------------------------------------------
# Constants – edit here if you keep the checkpoints in a custom path
# ----------------------------------------------------------------------

CHEMBERTA_NAME = "DeepChem/ChemBERTa-77M-MLM"          # 384-d
MOLFORMER_NAME = "ibm/MoLFormer-XL-both-10pct"         # 768-d

CHEMBERTA_DIM = 384
MOLFORMER_DIM = 768

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def canonical_smiles(s: str) -> str | None:
    """Canonicalise one SMILES string (same as RDKit in your notebook)."""
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


@torch.inference_mode()
def _embed_batch(smiles: List[str],
                 tokenizer,
                 model,
                 expected_dim: int,
                 device: torch.device) -> torch.Tensor:
    """
    Embed a list of SMILES and return a (n, dim) tensor on CPU.
    Pads / truncates silently if the output dim mismatches.
    """
    if not smiles:
        return torch.empty((0, expected_dim))

    inputs = tokenizer(
        smiles,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    outputs = model(**inputs)
    if outputs.pooler_output is not None:
        vecs = outputs.pooler_output         # (n, d)
    else:                                    # fallback: mean-pool last layer
        vecs = outputs.last_hidden_state.mean(1)

    if vecs.shape[1] != expected_dim:
        # rare, but protects you from odd checkpoints
        if vecs.shape[1] < expected_dim:                   # pad
            pad = expected_dim - vecs.shape[1]
            vecs = torch.nn.functional.pad(vecs, (0, pad))
        else:                                              # truncate
            vecs = vecs[:, :expected_dim]

    return vecs.cpu()


def _batched_embed(smiles: List[str],
                   tokenizer,
                   model,
                   dim: int,
                   batch_size: int,
                   device: torch.device) -> np.ndarray:
    """Embed arbitrarily long list of smiles with a progress bar."""
    all_vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(smiles), batch_size),
                  desc="⚙️  embedding",
                  leave=False):
        batch = smiles[i : i + batch_size]
        vecs = _embed_batch(batch, tokenizer, model, dim, device)
        all_vecs.append(vecs.numpy())

    return np.vstack(all_vecs)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def generate_embeddings(df: pd.DataFrame,
                        smiles_col: str = "can_smiles",
                        batch_size: int = 32,
                        device: str = "cpu") -> pd.DataFrame:
    """
    Return a new DataFrame with two array-columns:
        • chemberta_vec  – ndarray (shape = 384,)
        • molformer_vec  – ndarray (shape = 768,)
    The input df is *not* modified.
    """
    if smiles_col not in df.columns:
        raise KeyError(f"No '{smiles_col}' column found in df")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"💻  Using device: {device}")

    smiles = df[smiles_col].tolist()

    # --- ChemBERTa -----------------------------------------------------
    print("📦 Loading ChemBERTa-77 M…")
    cb_tok = AutoTokenizer.from_pretrained(CHEMBERTA_NAME, trust_remote_code=True)
    cb_model = AutoModel.from_pretrained(CHEMBERTA_NAME, trust_remote_code=True).to(device).eval()

    chemberta_arr = _batched_embed(smiles, cb_tok, cb_model,
                                   CHEMBERTA_DIM, batch_size, device)
    del cb_model; torch.cuda.empty_cache()

    # --- MolFormer -----------------------------------------------------
    print("📦 Loading MolFormer-XL…")
    mf_tok = AutoTokenizer.from_pretrained(MOLFORMER_NAME, trust_remote_code=True)
    mf_model = AutoModel.from_pretrained(MOLFORMER_NAME, trust_remote_code=True).to(device).eval()

    molformer_arr = _batched_embed(smiles, mf_tok, mf_model,
                                   MOLFORMER_DIM, batch_size, device)
    del mf_model; torch.cuda.empty_cache()

    out = df.copy()
    out["chemberta_vec"] = list(chemberta_arr)
    out["molformer_vec"] = list(molformer_arr)
    return out


def _save_outputs(df: pd.DataFrame,
                  prefix: str,
                  save_format: str = "parquet") -> None:
    """
    Save:
      • {prefix}.npz      – two big float32 matrices
      • {prefix}.<fmt>    – DataFrame with id, can_smiles, and cluster ints
    """
    vec_cb = np.stack(df["chemberta_vec"].values).astype("float32")
    vec_mf = np.stack(df["molformer_vec"].values).astype("float32")
    np.savez_compressed(f"{prefix}.npz",
                        chemberta=vec_cb,
                        molformer=vec_mf)

    # Replace the huge arrays by cheap indices before writing the table
    df_out = df.drop(columns=["chemberta_vec", "molformer_vec"])
    if save_format == "parquet":
        df_out.to_parquet(f"{prefix}.parquet", index=False)
    elif save_format == "feather":
        df_out.reset_index(drop=True).to_feather(f"{prefix}.feather")
    else:
        df_out.to_csv(f"{prefix}.csv", index=False)
    print(f"💾  Saved vectors → {prefix}.npz  |  metadata → {prefix}.{save_format}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate polymer embeddings")
    p.add_argument("--input", required=True,
                   help="CSV/Parquet file with at least a 'can_smiles' column")
    p.add_argument("--output", required=True,
                   help="Path prefix for output (no extension)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--save-format", choices=["parquet", "feather", "csv"],
                   default="parquet")
    return p


def main(argv: List[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    # 1) Read input
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    # ensure can_smiles exists (canonicalise on the fly if needed)
    if "can_smiles" not in df.columns:
        print("🔄  Canonicalising SMILES first…")
        df["can_smiles"] = df["SMILES"].map(canonical_smiles)

    # 2) Generate embeddings
    df_emb = generate_embeddings(df,
                                 batch_size=args.batch_size,
                                 device=args.device)

    # 3) Save
    _save_outputs(df_emb, args.output, args.save_format)


if __name__ == "__main__":
    main()
