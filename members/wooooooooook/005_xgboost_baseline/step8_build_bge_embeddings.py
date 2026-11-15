"""
step8_sample_news_pool.py가 캐싱해둔 기사 샘플을 BGE-M3(upskyy/bge-m3-korean, 004와 동일
모델, 파라미터 고정)로 임베딩해서 1024차원 벡터를 저장한다. 이게 attention의 key(가중치
계산용)로 쓰인다. step8_build_sentiment_raw.py와 같은 캐시 파일을 순서 그대로 읽어서
행 순서가 두 출력 사이에 정확히 일치한다.

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step8_build_bge_embeddings.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "features" / "news_sample_pool_step8.csv.gz"
OUT_PATH = BASE_DIR / "features" / "news_bge_embeddings_step8.npz"

MODEL_NAME = "upskyy/bge-m3-korean"
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


def main() -> None:
    df = pd.read_csv(SAMPLE_PATH)
    texts = df["text"].fillna("").astype(str).tolist()
    print(f"샘플 수: {len(texts)}")

    device = get_device()
    print(f"model={MODEL_NAME} device={device} (파라미터 고정, 추론 전용)")
    model = SentenceTransformer(MODEL_NAME, device=device)
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True,
    )
    print("임베딩 shape:", embeddings.shape)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        month=df["month"].to_numpy(),
        day=df["day"].to_numpy(),
        embeddings=embeddings.astype(np.float32),
    )
    print(f"저장 완료 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
