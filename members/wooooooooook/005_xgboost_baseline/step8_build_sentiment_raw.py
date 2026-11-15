"""
step8_sample_news_pool.py가 캐싱해둔 기사 샘플(step8_build_bge_embeddings.py와 동일한
파일, 동일한 행 순서)을 snunlp/KR-FinBert-SC(파라미터 고정)에 통과시켜 기사별 긍정/부정/
중립 확률을 저장한다. 월별로 집계하지 않고 기사 단위 그대로 저장 — attention의 value(집계
대상)로 쓰인다. 라벨 순서는 모델 config의 id2label을 읽어서 동적으로 매칭한다.

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step8_build_sentiment_raw.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "features" / "news_sample_pool_step8.csv.gz"
OUT_PATH = BASE_DIR / "features" / "news_sentiment_raw_step8.npz"

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


def _match_label(id2label: dict, keywords: list[str]) -> int:
    for idx, name in id2label.items():
        lower = name.lower()
        if any(k in lower for k in keywords):
            return idx
    raise ValueError(f"id2label에서 {keywords} 매칭 실패: {id2label}")


def main() -> None:
    df = pd.read_csv(SAMPLE_PATH)
    texts = df["text"].fillna("").astype(str).tolist()
    print(f"샘플 수: {len(texts)}")

    device = get_device()
    print(f"model={MODEL_NAME} device={device} (파라미터 고정, 추론 전용)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    id2label = model.config.id2label

    pos_idx = _match_label(id2label, ["pos", "긍정"])
    neg_idx = _match_label(id2label, ["neg", "부정"])
    neu_idx = _match_label(id2label, ["neu", "중립"])
    print(f"라벨 매핑: positive={pos_idx}, negative={neg_idx}, neutral={neu_idx} (id2label={id2label})")

    all_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="FinBERT 감성분석"):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        month=df["month"].to_numpy(),
        day=df["day"].to_numpy(),
        p_positive=probs[:, pos_idx].astype(np.float32),
        p_negative=probs[:, neg_idx].astype(np.float32),
        p_neutral=probs[:, neu_idx].astype(np.float32),
    )
    print(f"저장 완료 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
