"""
목적: 환율(원/달러)·국고채 3년/10년 금리·기준금리를 ECOS(한국은행 경제통계시스템)
OpenAPI로 일별 수집해 raw/daily_macro/{시리즈명}.csv로 저장. 015_daily_factor_panel의
fx_change(환율 변동률)·term_spread(국고채 10년-3년 금리차)·base_rate 일별 버전에 씀.
cpi_yoy(소비자물가)만 여기서 안 받음 — 통계청이 월 1회만 발표해서 구조적으로 일별화가
안 됨(raw/macro README 참고).

base_rate는 원래 "금통위 결정일만 알면 된다"고 별도 수집 안 하려 했는데, ECOS 통계표
722Y001이 일별(D) 주기도 지원해서 그냥 나머지와 똑같이 받기로 함(결정일 사이는 ECOS가
알아서 같은 값을 반복해줌 — 별도 forward-fill 로직 불필요).

통계표/항목 코드는 ECOS 사이트(https://ecos.bok.or.kr) 검색창에서 직접 확인함
(2026-11-21, 스크린샷으로 검증 완료).
"""
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLLECTOR_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "raw" / "daily_macro"
LOG_DIR = COLLECTOR_DIR / "log"

ECOS_API_KEY = None  # 환경변수 ECOS_API_KEY로 주입(아래 main에서 읽음)
BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"
FROM_DATE = "20000101"
TO_DATE = "20251231"
MAX_RETRIES = 3
BASE_BACKOFF_SEC = 5

SERIES = {
    "fx_usd_krw": {"stat_code": "731Y003", "cycle": "D", "item_code1": "0000003"},   # 원/달러(종가 15:30)
    # 0000013(그냥 "종가")은 2024-07-01부터만 데이터가 있어서(과거분 없음, 2026-11-21 확인)
    # 0000003(종가 15:30, 전통적으로 더 오래 쌓인 항목)로 교체함
    "treasury_3y": {"stat_code": "817Y002", "cycle": "D", "item_code1": "010200000"},  # 국고채(3년)
    "treasury_10y": {"stat_code": "817Y002", "cycle": "D", "item_code1": "010210000"},  # 국고채(10년)
    "base_rate": {"stat_code": "722Y001", "cycle": "D", "item_code1": "0101000"},      # 한국은행 기준금리
}

logger = logging.getLogger("collect_ecos_macro")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_DIR / "collect_ecos_macro.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(stream)


def fetch_series(name: str, spec: dict, api_key: str) -> pd.DataFrame | None:
    url = (f"{BASE_URL}/{api_key}/json/kr/1/100000/{spec['stat_code']}/"
           f"{spec['cycle']}/{FROM_DATE}/{TO_DATE}/{spec['item_code1']}")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "StatisticSearch" not in data:
                logger.warning(f"{name}: 응답에 StatisticSearch 없음 -> {data}")
                return None
            rows = data["StatisticSearch"]["row"]
            return pd.DataFrame(rows)
        except Exception as e:
            logger.warning(f"{name} 요청 실패 (시도 {attempt}/{MAX_RETRIES}): {type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(BASE_BACKOFF_SEC * (2 ** (attempt - 1)))
    return None


def main() -> None:
    import os

    setup_logging()
    api_key = os.environ.get("ECOS_API_KEY")
    if not api_key:
        raise SystemExit("ECOS_API_KEY 환경변수가 없습니다. https://ecos.bok.or.kr 에서 발급 후 "
                          "export ECOS_API_KEY=... 로 설정할 것.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in tqdm(SERIES.items(), desc="ECOS 매크로 수집"):
        out_path = OUTPUT_DIR / f"{name}.csv"
        if out_path.exists():
            continue
        df = fetch_series(name, spec, api_key)
        if df is None or df.empty:
            logger.warning(f"{name}: 수집 실패")
            continue
        df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"{name} 저장 완료 ({len(df)}행)")


if __name__ == "__main__":
    main()
