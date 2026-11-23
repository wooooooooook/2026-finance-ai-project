"""
목적: KOSPI 전종목의 일별 외국인 보유현황(상장주식수/보유수량/지분율/한도수량/한도소진률)을
pykrx로 받아 raw/daily_foreign_holding/{티커}.csv로 저장. raw/trading_flow/foreign_holding의
일별 버전(015_daily_factor_panel에서 씀).
"""
import getpass
import logging
import os
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
OUTPUT_DIR = PROJECT_ROOT / "raw" / "daily_foreign_holding"
LOG_DIR = COLLECTOR_DIR / "log"

FROM_DATE = "20000101"
TO_DATE = "20251231"
MAX_RETRIES = 3
BASE_BACKOFF_SEC = 5
REQUEST_TIMEOUT_SEC = 300

logger = logging.getLogger("collect_foreign_holding")
failed_logger = logging.getLogger("failed_tickers")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_DIR / "collect_foreign_holding.log", encoding="utf-8")
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
    for attempt in range(1, MAX_RETRIES + 1):
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                lambda: stock.get_exhaustion_rates_of_foreign_investment_by_date(FROM_DATE, TO_DATE, ticker)
            )
            return future.result(timeout=REQUEST_TIMEOUT_SEC)
        except FutureTimeoutError:
            logger.warning(f"{ticker} 요청 timeout ({REQUEST_TIMEOUT_SEC}초, 시도 {attempt}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES:
                failed_logger.info(f"{ticker}\tTimeoutError")
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


def _ensure_krx_credentials() -> None:
    """KRX_ID/KRX_PW 환경변수가 없으면 터미널에서 직접 입력받음
    (export로 넘기면 쉘 히스토리/프로세스 목록에 평문으로 남는 걸 피하기 위함)."""
    if not os.environ.get("KRX_ID"):
        os.environ["KRX_ID"] = input("KRX_ID: ").strip()
    if not os.environ.get("KRX_PW"):
        os.environ["KRX_PW"] = getpass.getpass("KRX_PW: ")


def main() -> None:
    _ensure_krx_credentials()
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_sorted_tickers()
    logger.info(f"총 {len(tickers)}개 종목, 수집 시작 ({FROM_DATE}~{TO_DATE})")

    done, failed, skipped = 0, 0, 0
    for ticker in tqdm(tickers, desc="일별 외국인 보유현황 수집"):
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
        logger.info(f"{ticker} 저장 완료 ({len(df)}행)")

        time.sleep(random.uniform(1, 3))

    logger.info(f"수집 종료 - 완료: {done}, 스킵(이미 있음): {skipped}, 실패: {failed}")
    if failed:
        logger.info(f"실패 목록: {LOG_DIR / 'failed_tickers.log'}")


if __name__ == "__main__":
    main()
