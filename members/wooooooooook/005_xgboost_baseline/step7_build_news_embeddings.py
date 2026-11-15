"""
step6_build_news_features.py와 같은 월별 일자가중 샘플(raw/news, 월 1,000개)을 재사용해서,
FinBERT(snunlp/KR-FinBert-SC, 파라미터 전부 고정 — 추론 전용)로 기사별 768차원 임베딩
([CLS] 토큰, 분류 헤드 통과 전 마지막 히든 상태)을 뽑아 저장한다.
step7_train_attention_pool.py의 입력으로 쓰인다.

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step7_build_news_embeddings.py
"""

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from step6_build_news_features import sample_all_months

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "features" / "news_embeddings.npz"

MODEL_NAME = "snunlp/KR-FinBert-SC"
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


def embed_articles(texts: list[str]) -> np.ndarray:
    device = get_device()
    print(f"model={MODEL_NAME} device={device} (파라미터 고정, 추론 전용)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # AutoModel: ...ForSequenceClassification 체크포인트에서 분류 헤드는 빼고 베이스
    # 인코더만 불러온다 (분류 헤드 weight는 안 맞는다는 경고가 뜨는데 정상임).
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    all_vecs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="FinBERT 임베딩 추출"):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            out = model(**enc)
            cls_vec = out.last_hidden_state[:, 0, :]  # [CLS] 토큰, 분류 헤드 통과 전
            all_vecs.append(cls_vec.cpu().numpy())

    return np.concatenate(all_vecs, axis=0)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sampled = sample_all_months()
    print(f"총 샘플 수: {len(sampled)}")

    embeddings = embed_articles(sampled["text"].fillna("").astype(str).tolist())
    print("임베딩 shape:", embeddings.shape)

    np.savez(
        OUT_PATH,
        month=sampled["month"].to_numpy(),
        day=sampled["day"].to_numpy(),
        embeddings=embeddings.astype(np.float32),
    )
    print(f"저장 완료 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
