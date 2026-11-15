"""
step6_build_news_features.py의 월별 일자가중 샘플링(raw/news, 월 1,000개)을 한 번만 돌려서
캐시로 저장한다. step8은 같은 기사에 대해 BGE-M3 임베딩(key)과 FinBERT 감성점수(value)를
각각 따로 추출해야 하는데, sample_all_months()는 멀티프로세스라 두 번 부르면 기사 순서가
보장되지 않는다. 그래서 한 번 뽑아 저장해두고 두 스크립트가 같은 캐시를 순서 그대로 읽게
해서 기사 단위로 정확히 정렬시킨다.

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step8_sample_news_pool.py
"""

from pathlib import Path

from step6_build_news_features import sample_all_months

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "features" / "news_sample_pool_step8.csv.gz"
SAMPLES_PER_MONTH = 2000  # step6/step7(월 1,000개)보다 늘려서 step8 전용으로 사용


def main() -> None:
    if OUT_PATH.exists():
        print(f"이미 존재함, 건너뜀: {OUT_PATH}")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sampled = sample_all_months(samples_per_month=SAMPLES_PER_MONTH)
    sampled = sampled.reset_index(drop=True)
    sampled.to_csv(OUT_PATH, index=False, compression="gzip")
    print(f"총 {len(sampled)}개 샘플 저장 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
