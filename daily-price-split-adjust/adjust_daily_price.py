"""
목적: raw/daily_price(수정 안 한 종가, adjusted=False)를 raw/returns의 adjusted_monthly_return과
같은 방식(액면분할·무상증자 등 자본변동을 반영한 수정주가)으로 보정해서
raw/daily_price_adjusted/{티커}.csv에 저장.

방법:
raw/daily_price에는 분할·무상증자 비율 정보가 없어서, raw/daily_marketcap의 '상장주식수'
일별 변화를 이용해 분할/무상증자 이벤트를 역산한다. 상장주식수가 하루 사이 크게(기본
5% 이상) 뛰고, 그 비율의 역수만큼 종가가 같은 날 반대로 움직이면(기계적 분할/무상증자의
특징) 그날을 분할 이벤트로 본다. 반대로 상장주식수는 늘었는데 종가가 그만큼 안 움직이면
유상증자(실제 자금이 들어온 경우)로 보고 보정하지 않는다 — 유상증자는 가격을 인위적으로
깨지 않기 때문.

이벤트가 감지되면 그 이벤트 "이전" 모든 날짜의 가격에 1/상장주식수비율을 누적해서 곱해
최신 날짜(raw 그대로) 기준으로 과거 가격을 역산한다(수정종가 = 최신 raw와 같은 스케일).
거래량·거래대금·등락률은 보정 대상이 아님(분할로 가격×주식수는 불변이라 원화 단위 값은
영향을 안 받음 — PREPROCESSING.md 근거 논의 참고). raw/daily_price 원본은 건드리지 않음.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOL_DIR = Path(__file__).resolve().parent
PRICE_DIR = PROJECT_ROOT / "raw" / "daily_price"
MARKETCAP_DIR = PROJECT_ROOT / "raw" / "daily_marketcap"
OUTPUT_DIR = PROJECT_ROOT / "raw" / "daily_price_adjusted"
LOG_DIR = TOOL_DIR / "log"

SHARE_JUMP_THRESHOLD = 0.05   # 상장주식수가 하루 사이 이 비율 이상 변하면 이벤트 후보
PRICE_INVERSE_TOLERANCE = 0.10  # 종가 변화가 상장주식수 변화의 역수와 이 오차 이내로 맞으면 분할로 확정
HALT_RESUME_TOLERANCE = 0.25  # 직전 거래일 거래량이 0(거래정지)이면 재개일 가격이 추가로 움직여서 허용오차를 넓힘(주식수·가격 방향이 반대일 때만). 1.0까지 넓히면 가격이 안 오른 감자까지 잡혀 월별과 어긋남
MIN_ROWS = 30  # 이보다 적으면(사실상 결측 종목, PREPROCESSING.md 7번) 건너뜀

PRICE_COLS = ["시가", "고가", "저가", "종가"]

logger = logging.getLogger("adjust_daily_price")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_DIR / "adjust_daily_price.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(stream)


def load_tickers() -> list[str]:
    return sorted(p.stem for p in PRICE_DIR.glob("*.csv"))


def adjust_ticker(ticker: str) -> tuple[pd.DataFrame | None, int]:
    """반환: (보정된 DataFrame 또는 None, 감지된 분할/무상증자 이벤트 수)."""
    price_path = PRICE_DIR / f"{ticker}.csv"
    mcap_path = MARKETCAP_DIR / f"{ticker}.csv"
    if not mcap_path.exists():
        logger.warning(f"{ticker}: daily_marketcap에 대응 파일 없음, 건너뜀")
        return None, 0

    price = pd.read_csv(price_path, dtype={"티커": str})
    if len(price) < MIN_ROWS:
        return None, 0

    mcap = pd.read_csv(mcap_path, dtype={"티커": str})[["날짜", "상장주식수"]]

    df = price.merge(mcap, on="날짜", how="inner").sort_values("날짜").reset_index(drop=True)
    if len(df) < MIN_ROWS:
        return None, 0

    shares = df["상장주식수"].to_numpy(dtype="float64")
    close = df["종가"].to_numpy(dtype="float64")
    n = len(df)

    shares_ratio = np.full(n, np.nan)
    shares_ratio[1:] = shares[1:] / shares[:-1]
    price_ratio = np.full(n, np.nan)
    price_ratio[1:] = np.divide(
        close[1:], close[:-1], out=np.full(n - 1, np.nan), where=close[:-1] != 0
    )

    is_event = np.zeros(n, dtype=bool)
    valid = ~np.isnan(shares_ratio) & ~np.isnan(price_ratio) & (shares_ratio > 0)
    share_jumped = np.abs(shares_ratio - 1) > SHARE_JUMP_THRESHOLD
    inverse_match = np.abs(shares_ratio * price_ratio - 1) < PRICE_INVERSE_TOLERANCE
    volume = df["거래량"].to_numpy(dtype="float64")
    after_halt = np.zeros(n, dtype=bool)
    after_halt[1:] = volume[:-1] == 0
    opposite_direction = (shares_ratio - 1) * (price_ratio - 1) < 0
    halt_resume_match = (
        after_halt & opposite_direction & (np.abs(shares_ratio * price_ratio - 1) < HALT_RESUME_TOLERANCE)
    )
    is_event[valid] = share_jumped[valid] & (inverse_match[valid] | halt_resume_match[valid])

    adj_factor = np.ones(n, dtype="float64")
    for i in range(n - 2, -1, -1):
        step = 1.0 / shares_ratio[i + 1] if is_event[i + 1] else 1.0
        adj_factor[i] = adj_factor[i + 1] * step

    out = price.merge(mcap[["날짜"]], on="날짜", how="inner").sort_values("날짜").reset_index(drop=True)
    for col in PRICE_COLS:
        out[f"수정{col}"] = df[col].to_numpy() * adj_factor
    out["수정등락률"] = out["수정종가"].pct_change(fill_method=None) * 100
    out["분할조정factor"] = adj_factor
    out["분할이벤트"] = is_event

    n_events = int(is_event.sum())
    return out, n_events


def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_tickers()
    logger.info(f"총 {len(tickers)}개 종목, 수정가격 재구성 시작")

    done, skipped, total_events = 0, 0, 0
    event_summary = []
    for ticker in tqdm(tickers, desc="daily_price 수정가격 재구성"):
        out_path = OUTPUT_DIR / f"{ticker}.csv"
        result, n_events = adjust_ticker(ticker)
        if result is None:
            skipped += 1
            continue
        result.to_csv(out_path, index=False, encoding="utf-8")
        done += 1
        total_events += n_events
        if n_events > 0:
            event_summary.append((ticker, n_events))
            logger.info(f"{ticker}: 분할/무상증자 이벤트 {n_events}건 감지, 저장 완료")

    logger.info(f"종료 - 완료: {done}, 건너뜀: {skipped}, 총 감지 이벤트: {total_events}건"
                f"({len(event_summary)}개 종목)")


if __name__ == "__main__":
    main()
