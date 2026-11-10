"""step2_embed_*.py 스크립트들이 공유하는 임베딩 실행 함수."""

import os

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "samples" / "sampled_pool.csv.gz"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

BATCH_SIZE = 32


def get_device() -> str:
    forced = os.environ.get("FORCE_DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_embedding(key: str, model_name: str) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EMBEDDINGS_DIR / f"{key}_embeddings.npz"
    if output_path.exists():
        print(f"이미 존재함, 건너뜀: {output_path}")
        return

    device = get_device()
    print(f"model={model_name} device={device}")

    df = pd.read_csv(SAMPLE_PATH)
    texts = df["text"].fillna("").astype(str).tolist()
    print(f"임베딩 대상: {len(texts)}개")

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    np.savez(
        output_path,
        titles=df["title"].fillna("").astype(str).to_numpy(dtype=object),
        years=df["year"].to_numpy(),
        embeddings=embeddings.astype(np.float32),
    )
    print(f"저장 완료 -> {output_path}")
