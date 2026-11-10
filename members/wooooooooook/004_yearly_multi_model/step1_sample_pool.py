"""
raw/news의 일별 csv.gz 전체를 연도 단위로 모은 뒤, 연도별로 5000개씩 무작위 추출(고정
시드)해서 제목+본문을 합친 텍스트를 하나의 캐시 파일(samples/sampled_pool.csv.gz)로
저장한다. 이 캐시를 embed_*.py 스크립트들이 공유해서 쓰므로, raw/news 전체를 모델마다
다시 읽을 필요가 없다.

연도마다 독립적인 작업이라 멀티프로세스로 병렬 처리한다 (NUM_WORKERS로 조절).

사용법:
    cd 004_yearly_multi_model
    uv run python sample_pool.py
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

NUM_WORKERS = os.cpu_count() or 4

BASE_DIR = Path(__file__).resolve().parent
RAW_NEWS_DIR = BASE_DIR.parent / "raw" / "news"
OUTPUT_PATH = BASE_DIR / "samples" / "sampled_pool.csv.gz"

YEARS = range(2001, 2021)
SAMPLES_PER_YEAR = 5000
RANDOM_STATE = 42


# 연도 하나를 처리한다: 그 해의 모든 일별 csv.gz를 풀링 후 5000개 무작위 샘플링.
# 별도 프로세스에서 실행된다.
def sample_year(year: int) -> pd.DataFrame:
    day_files = sorted(RAW_NEWS_DIR.glob(f"bigkinds_{year}-*.csv.gz"))
    frames = [pd.read_csv(f, usecols=["제목", "본문"]) for f in day_files]
    pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["제목", "본문"])

    rng = np.random.default_rng(RANDOM_STATE + year)
    n_sample = min(SAMPLES_PER_YEAR, len(pool))
    idx = rng.choice(len(pool), size=n_sample, replace=False)
    sampled = pool.iloc[idx].copy()
    sampled["year"] = year
    return sampled


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"연도 {len(list(YEARS))}개, {NUM_WORKERS}개 프로세스로 풀링+샘플링")

    results: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(sample_year, year): year for year in YEARS}
        for future in as_completed(futures):
            year = futures[future]
            df = future.result()
            print(f"  {year}: {len(df)}개 샘플링")
            results.append(df)

    combined = pd.concat(results, ignore_index=True).sort_values("year").reset_index(drop=True)

    titles = combined["제목"].fillna("").astype(str)
    bodies = combined["본문"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True)
    combined["text"] = (titles + ". " + bodies).str.strip()

    out = combined[["year", "제목", "text"]].rename(columns={"제목": "title"})
    out.to_csv(OUTPUT_PATH, index=False, compression="gzip")

    print(f"총 {len(out)}개 샘플 저장 -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
