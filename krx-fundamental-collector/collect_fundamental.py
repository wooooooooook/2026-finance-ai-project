"""
목적: KOSPI 전종목 PER/PBR/EPS/BPS/DIV/DPS를 월말 거래일마다 한 번씩 받아서
downloads/{YYYYMM}.csv로 저장. raw/trading_flow와 동일한 "파일 하나 = 그 달 전종목"
구조.

종목별로 1,086번 요청하는 대신, 월말 거래일(약 312개월)마다 전종목을 한 번에 주는
get_market_fundamental_by_ticker(date)를 써서 요청 수를 크게 줄임. 월말 거래일은
이미 받아둔 krx-daily-price-collector/downloads/000020.csv의 실제 거래일에서
추출함(000020은 pykrx 자체가 영업일 계산에 쓰는 기준 종목이기도 함).
"""
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

import pandas as pd
from pykrx import stock
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLLECTOR_DIR = Path(__file__).resolve().parent
CALENDAR_SOURCE = PROJECT_ROOT / "krx-daily-price-collector" / "downloads" / "000020.csv"
OUTPUT_DIR = COLLECTOR_DIR / "downloads"
LOG_DIR = COLLECTOR_DIR / "log"

MARKET = "KOSPI"
MAX_RETRIES = 3
BASE_BACKOFF_SEC = 5
REQUEST_TIMEOUT_SEC = 120

logger = logging.getLogger("collect_fundamental")
failed_logger = logging.getLogger("failed_months")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_DIR / "collect_fundamental.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(stream)

    failed_logger.setLevel(logging.INFO)
    failed_handler = logging.FileHandler(LOG_DIR / "failed_months.log", encoding="utf-8")
    failed_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    failed_logger.addHandler(failed_handler)


def load_month_end_dates() -> list[str]:
    """000020 실제 거래일에서 각 월의 마지막 거래일을 뽑음 (YYYYMMDD 문자열)."""
    if not CALENDAR_SOURCE.exists():
        raise FileNotFoundError(
            f"{CALENDAR_SOURCE} 가 없습니다. krx-daily-price-collector로 000020을 먼저 받아야 합니다."
        )
    df = pd.read_csv(CALENDAR_SOURCE, usecols=["날짜"], parse_dates=["날짜"])
    month_end = df.groupby(df["날짜"].dt.to_period("M"))["날짜"].max()
    return sorted(d.strftime("%Y%m%d") for d in month_end)


def fetch_with_retry(date: str) -> pd.DataFrame | None:
    for attempt in range(1, MAX_RETRIES + 1):
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(stock.get_market_fundamental_by_ticker, date, MARKET)
            return future.result(timeout=REQUEST_TIMEOUT_SEC)
        except FutureTimeoutError:
            logger.warning(f"{date} 요청 timeout ({REQUEST_TIMEOUT_SEC}초, 시도 {attempt}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES:
                failed_logger.info(f"{date}\tTimeoutError")
                return None
            time.sleep(BASE_BACKOFF_SEC * (2 ** (attempt - 1)))
        except Exception as e:
            logger.warning(f"{date} 요청 실패 (시도 {attempt}/{MAX_RETRIES}): {type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                failed_logger.info(f"{date}\t{type(e).__name__}: {e}")
                return None
            time.sleep(BASE_BACKOFF_SEC * (2 ** (attempt - 1)))
        finally:
            executor.shutdown(wait=False)
    return None


def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    month_end_dates = load_month_end_dates()
    logger.info(f"총 {len(month_end_dates)}개월, 수집 시작 ({month_end_dates[0]}~{month_end_dates[-1]})")

    done, failed, skipped = 0, 0, 0
    for date in tqdm(month_end_dates, desc="펀더멘털 수집(월별)"):
        ym = date[:6]
        out_path = OUTPUT_DIR / f"{ym}.csv"
        if out_path.exists():
            skipped += 1
            continue

        df = fetch_with_retry(date)
        if df is None or df.empty:
            failed += 1
            if df is not None and df.empty:
                failed_logger.info(f"{date}\tEmptyResult")
            time.sleep(random.uniform(1, 2))
            continue

        df = df.reset_index()
        df.insert(0, "기준일", date)
        df.to_csv(out_path, index=False, encoding="utf-8")
        done += 1
        logger.info(f"{ym} 저장 완료 ({len(df)}개 종목, 기준일 {date})")

        time.sleep(random.uniform(1, 2))

    logger.info(f"수집 종료 - 완료: {done}, 스킵(이미 있음): {skipped}, 실패: {failed}")
    if failed:
        logger.info(f"실패 목록: {LOG_DIR / 'failed_months.log'}")


if __name__ == "__main__":
    main()
