"""
목적: raw/returns 표본 1,086종목의 일별 가격(OHLCV)을 pykrx로 수집해 downloads/{티커}.csv로 저장.

adjusted=False로 받는다. adjusted=True(네이버 소스)는 "오늘 기준 최근 N일"만 요청하는
방식이라 fromdate를 아무리 과거로 줘도 서버가 최근 약 11년치만 돌려줘서 2000년대 데이터를
받을 수 없었음(실측 확인). adjusted=False는 KRX 소스를 쓰며 fromdate/todate를 그대로
서버에 보내는 실제 날짜범위 조회라 이 문제가 없음. 대신 액면분할 등 자본변동이 반영되지
않은 raw 종가라는 점은 감안해야 함(분할 시점에 가격이 끊겨 보일 수 있음).
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
TICKER_SOURCE = PROJECT_ROOT / "raw" / "returns" / "train_2001_2021__adjusted_monthly_return.csv"
OUTPUT_DIR = COLLECTOR_DIR / "downloads"
LOG_DIR = COLLECTOR_DIR / "log"

FROM_DATE = "20000101"
TO_DATE = "20251231"
MAX_RETRIES = 3
BASE_BACKOFF_SEC = 5
REQUEST_TIMEOUT_SEC = 300

logger = logging.getLogger("collect_daily_price")
failed_logger = logging.getLogger("failed_tickers")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_DIR / "collect_daily_price.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(stream)

    failed_logger.setLevel(logging.INFO)
    failed_handler = logging.FileHandler(LOG_DIR / "failed_tickers.log", encoding="utf-8")
    failed_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    failed_logger.addHandler(failed_handler)


def load_sorted_tickers() -> list[str]:
    header = pd.read_csv(TICKER_SOURCE, nrows=0, dtype=str)
    tickers = [c for c in header.columns if c != "month"]
    return sorted(tickers)


def fetch_with_retry(ticker: str) -> pd.DataFrame | None:
    # pykrx 내부에는 이 요청(개별종목시세)에 timeout이 걸려 있지 않아서, 서버가 응답을
    # 안 주면 스크립트가 영원히 멈출 수 있음. 그래서 별도 스레드로 돌려서 직접 시간 제한을 둠.
    for attempt in range(1, MAX_RETRIES + 1):
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                stock.get_market_ohlcv_by_date, FROM_DATE, TO_DATE, ticker, adjusted=False
            )
            return future.result(timeout=REQUEST_TIMEOUT_SEC)
        except FutureTimeoutError:
            logger.warning(
                f"{ticker} 요청 timeout ({REQUEST_TIMEOUT_SEC}초 초과, 시도 {attempt}/{MAX_RETRIES})"
            )
            if attempt == MAX_RETRIES:
                failed_logger.info(f"{ticker}\tTimeoutError: {REQUEST_TIMEOUT_SEC}초 초과")
                return None
            time.sleep(BASE_BACKOFF_SEC * (2 ** (attempt - 1)))
        except Exception as e:
            logger.warning(f"{ticker} 요청 실패 (시도 {attempt}/{MAX_RETRIES}): {type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                failed_logger.info(f"{ticker}\t{type(e).__name__}: {e}")
                return None
            time.sleep(BASE_BACKOFF_SEC * (2 ** (attempt - 1)))
        finally:
            executor.shutdown(wait=False)
    return None


def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_sorted_tickers()
    logger.info(f"총 {len(tickers)}개 종목, 수집 시작 ({FROM_DATE}~{TO_DATE}, adjusted=False)")

    done, failed, skipped = 0, 0, 0
    for ticker in tqdm(tickers, desc="일별 가격 수집"):
        out_path = OUTPUT_DIR / f"{ticker}.csv"
        if out_path.exists():
            skipped += 1
            continue

        df = fetch_with_retry(ticker)
        if df is None or df.empty:
            failed += 1
            if df is not None and df.empty:
                failed_logger.info(f"{ticker}\tEmptyResult")
            time.sleep(random.uniform(1, 3))
            continue

        df = df.reset_index()
        df.insert(0, "티커", ticker)
        df.to_csv(out_path, index=False, encoding="utf-8")
        done += 1
        logger.info(f"{ticker} 저장 완료 ({len(df)}행, {df['날짜'].min()}~{df['날짜'].max()})")

        time.sleep(random.uniform(1, 3))

    logger.info(f"수집 종료 - 완료: {done}, 스킵(이미 있음): {skipped}, 실패: {failed}")
    if failed:
        logger.info(f"실패 목록: {LOG_DIR / 'failed_tickers.log'}")


if __name__ == "__main__":
    main()
