"""
bigkinds-crawler/downloads/ 에 있는 일별 xlsx 전체(한 달 전체 일자)를 월 단위로 모은 뒤,
월별로 3000개씩 무작위 추출(고정 시드)하고 '제목'+'본문'을 합쳐
snunlp/KR-SBERT-V40K-klueNLI-augSTS 모델로 768차원 임베딩을 만들어 저장한다.
(기존 "매월 15일 1개 파일만" 방식 대신, 해당 월의 모든 일자 데이터를 풀로 사용)
"""

import re

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
DOWNLOADS_DIR = Path(__file__).parent.parent / "bigkinds-crawler" / "downloads"
OUTPUT_DIR = Path(__file__).parent / "embeddings"
YEARS = range(2001, 2006)
SAMPLES_PER_MONTH = 3000
BATCH_SIZE = 64
RANDOM_STATE = 42

DAY_FILE_PATTERN = re.compile(r"bigkinds_(\d{4})-(\d{2})-\d{2}\.xlsx$")


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_month_pool(year: int, month: int) -> pd.DataFrame:
    day_files = sorted(DOWNLOADS_DIR.glob(f"bigkinds_{year}-{month:02d}-*.xlsx"))
    frames = []
    for day_file in day_files:
        df = pd.read_excel(day_file, usecols=["제목", "본문"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["제목", "본문"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    rng = np.random.default_rng(RANDOM_STATE)

    months = [(year, month) for year in YEARS for month in range(1, 13)]

    for year, month in tqdm(months, desc="months"):
        output_path = OUTPUT_DIR / f"bigkinds_{year}-{month:02d}_sampled_embeddings.npz"
        if output_path.exists():
            continue

        pool = load_month_pool(year, month)
        n_pool = len(pool)
        n_sample = min(SAMPLES_PER_MONTH, n_pool)
        print(f"{year}-{month:02d}: pool={n_pool}, sampling={n_sample}")

        if n_pool == 0:
            continue

        idx = rng.choice(n_pool, size=n_sample, replace=False)
        sampled = pool.iloc[idx]

        titles = sampled["제목"].fillna("").astype(str)
        bodies = sampled["본문"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True)
        combined = (titles + ". " + bodies).str.strip().tolist()

        embeddings = model.encode(
            combined,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        np.savez(
            output_path,
            titles=np.array(titles.tolist(), dtype=object),
            embeddings=embeddings.astype(np.float32),
        )

    print(f"done. outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
