"""
목적: KOSPI 전종목의 투자자별(개인/외국인 등) 일별 매수·매도 거래대금을 pykrx로 수집해
raw/daily_investor_flow/{티커}.csv로 저장. raw/trading_flow/net_individual·net_foreign의
일별 버전(015_daily_factor_panel에서 씀).

pykrx의 get_market_trading_value_by_date는 on="매수"/"매도"/"순매수" 중 **하나만** 돌려준다.
AGENTS.md의 거래강도(매수+매도)·순매수강도(매수-매도) 공식을 계산하려면 매수·매도 금액이
각각 따로 있어야 해서, 종목 하나당 "매수" 조회 1번 + "매도" 조회 1번, 총 2번 요청한다
(krx-daily-price-collector보다 요청 수가 2배).

컬럼은 pykrx가 돌려주는 이름(기관합계/기타법인/개인/외국인합계/전체)을 그대로 쓰고
_매수거래대금/_매도거래대금 접미사만 붙인다 — raw는 가공하지 않고 그대로 보관.
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
OUTPUT_DIR = PROJECT_ROOT / "raw" / "daily_investor_flow"
LOG_DIR = COLLECTOR_DIR / "log"

FROM_DATE = "20000101"
TO_DATE = "20251231"
MAX_RETRIES = 3
BASE_BACKOFF_SEC = 5
REQUEST_TIMEOUT_SEC = 300

logger = logging.getLogger("collect_investor_flow")
failed_logger = logging.getLogger("failed_tickers")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_DIR / "collect_investor_flow.log", encoding="utf-8")
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


def _fetch_on(ticker: str, on: str) -> pd.DataFrame | None:
    for attempt in range(1, MAX_RETRIES + 1):
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                lambda: stock.get_market_trading_value_by_date(FROM_DATE, TO_DATE, ticker, on=on)
            )
            return future.result(timeout=REQUEST_TIMEOUT_SEC)
        except FutureTimeoutError:
            logger.warning(f"{ticker}({on}) 요청 timeout ({REQUEST_TIMEOUT_SEC}초, 시도 {attempt}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES:
                failed_logger.info(f"{ticker}\t{on}\tTimeoutError")
                return None
            time.sleep(BASE_BACKOFF_SEC * (2 ** (attempt - 1)))
        except Exception as e:
            logger.warning(f"{ticker}({on}) 요청 실패 (시도 {attempt}/{MAX_RETRIES}): {type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                failed_logger.info(f"{ticker}\t{on}\t{type(e).__name__}: {e}")
                return None
            time.sleep(BASE_BACKOFF_SEC * (2 ** (attempt - 1)))
        finally:
            executor.shutdown(wait=False)
    return None


def fetch_ticker(ticker: str) -> pd.DataFrame | None:
    buy = _fetch_on(ticker, "매수")
    if buy is None or buy.empty:
        return None
    time.sleep(random.uniform(1, 2))
    sell = _fetch_on(ticker, "매도")
    if sell is None or sell.empty:
        return None

    df = buy.add_suffix("_매수거래대금").join(sell.add_suffix("_매도거래대금"), how="outer")
    return df


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
    logger.info(f"총 {len(tickers)}개 종목, 수집 시작 ({FROM_DATE}~{TO_DATE}, 종목당 매수/매도 2회 조회)")

    done, failed, skipped = 0, 0, 0
    for ticker in tqdm(tickers, desc="일별 투자자 거래대금 수집"):
        out_path = OUTPUT_DIR / f"{ticker}.csv"
        if out_path.exists():
            skipped += 1
            continue

        df = fetch_ticker(ticker)
        if df is None or df.empty:
            failed += 1
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
