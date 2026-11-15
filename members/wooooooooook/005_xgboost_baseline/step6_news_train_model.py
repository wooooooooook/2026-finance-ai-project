"""
정형 피처(매매동향 4종 + 수익률 lag/타겟)를 만들고, XGBoost로 t+1월 수익률을 학습·검증·
평가한다.

- build_trading_flow_features(): raw/trading_flow(net_individual, net_foreign) + raw/returns
  (시가총액)를 조인해 AGENTS.md의 4개 매매동향 피처 계산
- build_return_features(): raw/returns(adjusted_monthly_return)로 수익률 lag(1·3·6·12개월,
  3/6/12는 누적 복리수익률) + 다음달(t+1) 타겟 계산
- train / validate / evaluate: train(2001-2017)으로 학습, validation(2018-2021)으로 성능
  확인, test(2022-2025)는 최종 1회만 평가

- load_macro_features(): raw/macro(train.csv, test.csv)의 5개 매크로 변수(base_rate,
  fx_change, term_spread, cpi_yoy, kospi_rv)를 월 기준으로 그대로 붙임 (종목 무관, 같은
  달이면 전 종목에 동일 값)

- load_news_features(): step6_build_news_features.py가 만든 뉴스 감성 피처(월당 3개:
  sentiment_positive/negative/neutral, 일자 가중평균)를 월 기준으로 그대로 붙임 (종목
  무관, 매크로와 동일한 방식)

step4_baseline_train_model.py(매매동향+매크로)에 뉴스 피처를 추가한 버전.
step6_build_news_features.py를 먼저 실행해서 features/news_features.csv를 만들어둬야 한다.

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step6_news_train_model.py
"""

import csv
from pathlib import Path

import pandas as pd
import xgboost as xgb
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
RAW = PROJECT_ROOT / "raw"
FEATURES_DIR = BASE_DIR / "features"

TRAIN_YEARS = ("2001-01", "2017-12")
VAL_YEARS = ("2018-01", "2021-12")
TEST_YEARS = ("2022-01", "2025-12")


# ---------------------------------------------------------------------------
# 1) 매매동향 피처 (개인/외국인 순매수강도·거래강도)
# ---------------------------------------------------------------------------

MARKET_CAP_FILES = [
    RAW / "returns" / "train_2001_2021__raw_month_end_market_cap.csv",
    RAW / "returns" / "test_2022_2025__raw_month_end_market_cap.csv",
]


def _load_market_cap():
    out = {}
    for path in MARKET_CAP_FILES:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            codes = next(reader)[1:]
            for row in reader:
                month = row[0]
                for code, val in zip(codes, row[1:]):
                    out[(month, code)] = float(val) if val else None
    return out


def _load_flow_month(folder, yyyymm):
    path = RAW / "trading_flow" / folder / f"{yyyymm}.csv"
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["티커"]] = (float(row["매도거래대금"]), float(row["매수거래대금"]))
    return out


def build_trading_flow_features() -> Path:
    out_path = FEATURES_DIR / "trading_flow_features.csv"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    market_cap = _load_market_cap()
    months = sorted({ym for ym, _ in market_cap})
    print(f"[trading_flow] 대상 월 수: {len(months)} ({months[0]} ~ {months[-1]})")

    n_rows = 0
    n_foreign_missing = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "month", "code",
            "individual_net_intensity", "individual_trade_intensity",
            "foreign_net_intensity", "foreign_trade_intensity",
        ])
        for month in tqdm(months, desc="trading_flow 피처"):
            yyyymm = month.replace("-", "")
            individual = _load_flow_month("net_individual", yyyymm)
            foreign = _load_flow_month("net_foreign", yyyymm)

            for code in (c for (m, c) in market_cap if m == month):
                mktcap = market_cap.get((month, code))
                if not mktcap:
                    continue

                if code in individual:
                    sell, buy = individual[code]
                    ind_net, ind_trd = (buy - sell) / mktcap, (buy + sell) / mktcap
                else:
                    ind_net = ind_trd = ""

                if code in foreign:
                    sell, buy = foreign[code]
                    for_net, for_trd = (buy - sell) / mktcap, (buy + sell) / mktcap
                else:
                    for_net = for_trd = ""
                    n_foreign_missing += 1

                w.writerow([month, code, ind_net, ind_trd, for_net, for_trd])
                n_rows += 1

    print(f"[trading_flow] 저장 완료 -> {out_path}, 행수={n_rows}, "
          f"외국인 결측={n_foreign_missing / n_rows:.1%}")
    return out_path


# ---------------------------------------------------------------------------
# 2) 수익률 lag 피처 + 타겟
# ---------------------------------------------------------------------------

RETURN_FILES = [
    RAW / "returns" / "warmup_2000__adjusted_monthly_return.csv",
    RAW / "returns" / "train_2001_2021__adjusted_monthly_return.csv",
    RAW / "returns" / "test_2022_2025__adjusted_monthly_return.csv",
]
OUTPUT_RETURN_FILES = [
    RAW / "returns" / "train_2001_2021__adjusted_monthly_return.csv",
    RAW / "returns" / "test_2022_2025__adjusted_monthly_return.csv",
]


def _load_returns(paths):
    out = {}
    for path in paths:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            codes = next(reader)[1:]
            for row in reader:
                month = row[0]
                for code, val in zip(codes, row[1:]):
                    out[(month, code)] = float(val) if val else None
    return out


def _shift_ym(ym, months):
    y, m = int(ym[:4]), int(ym[5:7])
    total = y * 12 + (m - 1) - months
    ny, nm = divmod(total, 12)
    return f"{ny:04d}-{nm + 1:02d}"


def _cum_return(returns, code, month, window):
    prod = 1.0
    for k in range(1, window + 1):
        r = returns.get((_shift_ym(month, k), code))
        if r is None:
            return None
        prod *= (1 + r)
    return prod - 1


def build_return_features() -> Path:
    out_path = FEATURES_DIR / "return_features.csv"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    returns = _load_returns(RETURN_FILES)
    output_months = sorted({
        row[0]
        for path in OUTPUT_RETURN_FILES
        for row in csv.reader(open(path, encoding="utf-8-sig"))
    } - {"month"})
    codes = sorted({code for (_, code) in returns})
    print(f"[returns] 대상 월 수: {len(output_months)} ({output_months[0]} ~ {output_months[-1]}), "
          f"종목 수: {len(codes)}")

    n_rows = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["month", "code", "return_lag1", "return_lag3", "return_lag6", "return_lag12", "target_next"])
        for month in tqdm(output_months, desc="return 피처"):
            next_month = _shift_ym(month, -1)
            for code in codes:
                if (month, code) not in returns:
                    continue
                lag1 = returns.get((_shift_ym(month, 1), code))
                lag3 = _cum_return(returns, code, month, 3)
                lag6 = _cum_return(returns, code, month, 6)
                lag12 = _cum_return(returns, code, month, 12)
                target = returns.get((next_month, code))
                w.writerow([
                    month, code,
                    lag1 if lag1 is not None else "",
                    lag3 if lag3 is not None else "",
                    lag6 if lag6 is not None else "",
                    lag12 if lag12 is not None else "",
                    target if target is not None else "",
                ])
                n_rows += 1

    print(f"[returns] 저장 완료 -> {out_path}, 행수={n_rows}")
    return out_path


# ---------------------------------------------------------------------------
# 3) 매크로 피처 (종목 무관, 월 단위로 그대로 붙임)
# ---------------------------------------------------------------------------

MACRO_COLS = ["base_rate", "fx_change", "term_spread", "cpi_yoy", "kospi_rv"]
MACRO_FILES = [RAW / "macro" / "train.csv", RAW / "macro" / "test.csv"]


def load_macro_features() -> pd.DataFrame:
    macro = pd.concat([pd.read_csv(p, dtype={"ym": str}) for p in MACRO_FILES], ignore_index=True)
    macro["month"] = macro["ym"].str[:4] + "-" + macro["ym"].str[4:]
    return macro[["month"] + MACRO_COLS]


# ---------------------------------------------------------------------------
# 4) 뉴스 감성 피처 (종목 무관, 월 단위로 그대로 붙임)
# ---------------------------------------------------------------------------

NEWS_COLS = ["sentiment_positive", "sentiment_negative", "sentiment_neutral"]
NEWS_FILE = FEATURES_DIR / "news_features.csv"


def load_news_features() -> pd.DataFrame:
    news = pd.read_csv(NEWS_FILE)
    return news[["month"] + NEWS_COLS]


# ---------------------------------------------------------------------------
# 5) 학습 (XGBoost) — train만 사용해서 학습하고 모델 weight를 저장한다.
#    validation/test 평가는 step6_news_evaluate.py에서 따로 한다.
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "return_lag1", "return_lag3", "return_lag6", "return_lag12",
    "individual_net_intensity", "individual_trade_intensity",
    "foreign_net_intensity", "foreign_trade_intensity",
] + MACRO_COLS + NEWS_COLS
MODEL_PATH = BASE_DIR / "models" / "xgb_news.json"


def load_merged_features():
    # code는 "0120G0"처럼 영문이 섞인 종목코드가 있어서 dtype을 명시하지 않으면 pandas가
    # 청크별로 다르게 추론해(DtypeWarning) 같은 컬럼 안에 int/str이 섞이고, merge 키가
    # 서로 안 맞아 매매동향 피처가 거의 다 매칭 실패하는 버그가 있었다. 반드시 str로 고정.
    trading = pd.read_csv(FEATURES_DIR / "trading_flow_features.csv", dtype={"code": str})
    returns = pd.read_csv(FEATURES_DIR / "return_features.csv", dtype={"code": str})
    df = returns.merge(trading, on=["month", "code"], how="left")
    df = df.merge(load_macro_features(), on="month", how="left")
    df = df.merge(load_news_features(), on="month", how="left")
    return df.dropna(subset=["target_next"])  # 타겟 없는 행(2025-12 등)은 제외


class _TqdmBoostCallback(xgb.callback.TrainingCallback):
    """부스팅 라운드(n_estimators)마다 진행률을 표시하는 XGBoost 콜백."""

    def __init__(self, total):
        super().__init__()
        self.pbar = tqdm(total=total, desc="xgboost 학습")

    def after_training(self, model):
        self.pbar.close()
        return model

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False


def train_and_save():
    df = load_merged_features()
    train_df = df[(df["month"] >= TRAIN_YEARS[0]) & (df["month"] <= TRAIN_YEARS[1])]
    print(f"train={len(train_df)}")

    n_estimators = 500
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        missing=float("nan"),
        callbacks=[_TqdmBoostCallback(n_estimators)],
    )
    model.fit(train_df[FEATURE_COLS], train_df["target_next"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"모델 저장 완료 -> {MODEL_PATH}")
    return model


def main() -> None:
    build_trading_flow_features()
    build_return_features()
    train_and_save()


if __name__ == "__main__":
    main()
