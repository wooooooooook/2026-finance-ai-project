"""
raw/news에서 월별로 일자별 뉴스량 비중에 맞춰 ~1,000개씩 샘플링하고, snunlp/KR-FinBert-SC로
긍정/부정/중립 확률을 구한 뒤, 월말에 가까운 뉴스일수록 더 크게 반영하는 날짜 가중평균으로
월별 피처(월당 3개 컬럼)를 만든다.

- 샘플링: 그 달의 일별 기사 수 비중대로 하루당 샘플 수를 배분(평일에 많이 나오면 평일을
  더 많이 뽑음), 총 SAMPLES_PER_MONTH개 목표
- 감성분석: snunlp/KR-FinBert-SC. 라벨 순서를 하드코딩하지 않고 모델 config의 id2label을
  읽어서 긍정/부정/중립에 매칭
- 날짜 가중치: weight(day) = day_of_month / days_in_month (선형, 월말에 가까울수록 큼)
- 출력: features/news_features.csv (month, sentiment_positive, sentiment_negative,
  sentiment_neutral)

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step6_build_news_features.py
"""

import calendar
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
RAW_NEWS_DIR = PROJECT_ROOT / "raw" / "news"
OUT_PATH = BASE_DIR / "features" / "news_features.csv"

NUM_WORKERS = os.cpu_count() or 4
SAMPLES_PER_MONTH = 1000
RANDOM_STATE = 42

MODEL_NAME = "snunlp/KR-FinBert-SC"
BATCH_SIZE = 32

YEARS = range(2001, 2026)
MONTHS_ALL = [f"{y:04d}-{m:02d}" for y in YEARS for m in range(1, 13)]


def sample_month(month: str, samples_per_month: int = SAMPLES_PER_MONTH) -> pd.DataFrame:
    """그 달 일별 기사 수 비중대로 하루당 샘플 수를 배분해서 뽑는다."""
    day_files = sorted(RAW_NEWS_DIR.glob(f"bigkinds_{month}-*.csv.gz"))
    if not day_files:
        return pd.DataFrame(columns=["month", "day", "text"])

    rng = np.random.default_rng(RANDOM_STATE + int(month.replace("-", "")))

    day_pools = []
    day_counts = []
    for f in day_files:
        # f.stem은 .gz만 벗겨내서 "bigkinds_YYYY-MM-DD.csv"가 됨 -> .csv까지 마저 제거
        date_str = f.name.removesuffix(".csv.gz")  # bigkinds_YYYY-MM-DD
        day = int(date_str.split("-")[-1])
        pool = pd.read_csv(f, usecols=["제목", "본문"])
        day_pools.append((day, pool))
        day_counts.append(len(pool))

    total = sum(day_counts)
    if total == 0:
        return pd.DataFrame(columns=["month", "day", "text"])

    rows = []
    for (day, pool), n_day in zip(day_pools, day_counts):
        if n_day == 0:
            continue
        quota = round(samples_per_month * n_day / total)
        quota = min(quota, n_day)
        if quota == 0:
            continue
        idx = rng.choice(n_day, size=quota, replace=False)
        sampled = pool.iloc[idx]
        titles = sampled["제목"].fillna("").astype(str)
        bodies = sampled["본문"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True)
        text = (titles + ". " + bodies).str.strip()
        rows.append(pd.DataFrame({"month": month, "day": day, "text": text}))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["month", "day", "text"])


def sample_all_months(samples_per_month: int = SAMPLES_PER_MONTH) -> pd.DataFrame:
    print(f"{len(MONTHS_ALL)}개월, {NUM_WORKERS}개 프로세스로 일자 비중 샘플링 (월 {samples_per_month}개 목표)")
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(sample_month, m, samples_per_month): m for m in MONTHS_ALL}
        with tqdm(total=len(futures), desc="월별 샘플링") as pbar:
            for future in as_completed(futures):
                df = future.result()
                pbar.update(1)
                results.append(df)
    return pd.concat(results, ignore_index=True)


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


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    device = get_device()
    print(f"model={MODEL_NAME} device={device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()
    id2label = model.config.id2label

    pos_idx = _match_label(id2label, ["pos", "긍정"])
    neg_idx = _match_label(id2label, ["neg", "부정"])
    neu_idx = _match_label(id2label, ["neu", "중립"])
    print(f"라벨 매핑: positive={pos_idx}, negative={neg_idx}, neutral={neu_idx} (id2label={id2label})")

    texts = df["text"].fillna("").astype(str).tolist()
    all_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="FinBERT 감성분석"):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    df = df.copy()
    df["p_positive"] = probs[:, pos_idx]
    df["p_negative"] = probs[:, neg_idx]
    df["p_neutral"] = probs[:, neu_idx]
    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    def _days_in_month(month: str) -> int:
        y, m = int(month[:4]), int(month[5:7])
        return calendar.monthrange(y, m)[1]

    df = df.copy()
    df["days_in_month"] = df["month"].map(_days_in_month)
    df["weight"] = df["day"] / df["days_in_month"]  # 월말에 가까울수록 큰 선형 가중치

    def _weighted_avg(g: pd.DataFrame) -> pd.Series:
        w = g["weight"]
        return pd.Series({
            "sentiment_positive": np.average(g["p_positive"], weights=w),
            "sentiment_negative": np.average(g["p_negative"], weights=w),
            "sentiment_neutral": np.average(g["p_neutral"], weights=w),
        })

    return df.groupby("month").apply(_weighted_avg, include_groups=False).reset_index()


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sampled = sample_all_months()
    print(f"총 샘플 수: {len(sampled)}")

    scored = run_sentiment(sampled)
    monthly = aggregate_monthly(scored)

    monthly.to_csv(OUT_PATH, index=False)
    print(f"저장 완료 -> {OUT_PATH}, 행수={len(monthly)}")


if __name__ == "__main__":
    main()
