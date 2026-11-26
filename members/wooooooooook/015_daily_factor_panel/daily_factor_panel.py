"""
daily_factor_panel.py: 015 실험 — 006/007/013(월별 23-factor baseline)을 거래일 단위로
재현한다. 종목×거래일 feature 패널을 만들고, 단일 팩터 Rank IC와 XGBoost 전체모델
Rank IC를 월별 baseline과 비교한다. (실험 내용은 README.md 참고)

월별(006) -> 일별 치환:
  raw/returns(adjusted_monthly_return 등)      -> raw/daily_price_adjusted(수정종가)
  raw/trading_flow/price_change(거래대금)       -> raw/daily_marketcap(거래대금·시가총액·상장주식수)
  raw/trading_flow/net_individual·net_foreign  -> raw/daily_investor_flow(개인·외국인합계 매수/매도거래대금)
  raw/trading_flow/foreign_holding             -> raw/daily_foreign_holding(상장주식수·보유수량·한도소진률)
  raw/macro(월별, 이미 lag 적용)                -> raw/daily_macro(일별) + raw/macro(cpi_yoy만, 월 그대로 전파)

1개월/3개월/12개월 rolling window는 거래일 수로 근사(21/63/252일) — 정확한 달력월이
아니라 "약 1/3/12개월"이라는 점에 유의(README 참고).

단계 (--stages로 골라서 실행, 기본은 전체):
  features   원본 -> 종목×거래일 feature 패널 (features/panel.csv.gz). 이미 있으면 재사용(--rebuild로 강제)
  factor_ic  모델 없이 feature 하나씩 다음 거래일 수익률과의 일별 Rank IC (train/val)
  train      전체 feature로 XGBoost 학습 -> models/xgb_all.json
  evaluate   저장된 모델 평가 (전체 지표 + 연도별 IC, 006/007과 비교)

사전조건: daily-price-split-adjust/adjust_daily_price.py를 먼저 실행해서
raw/daily_price_adjusted/가 존재해야 함.

사용법:
    cd members/wooooooooook/015_daily_factor_panel
    python3 daily_factor_panel.py                               # 전체 단계
    python3 daily_factor_panel.py --stages features factor_ic   # 일부 단계만
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

# ===========================================================================
# 설정
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
RAW = PROJECT_ROOT / "raw"

FEATURES_PATH = BASE_DIR / "features" / "panel.csv.gz"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "xgb_all.json"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"

# 006과 동일한 연도 분할(일 단위로 변환)
SPLITS = {
    "train": ("2001-01-01", "2017-12-31"),
    "val": ("2018-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2025-12-31"),
}
OUTPUT_START, OUTPUT_END = "2001-01-01", "2025-12-31"

# PREPROCESSING.md 7번: 사실상 전 기간 결측인 종목 (daily_price/daily_investor_flow/
# daily_foreign_holding/daily_marketcap 전부 동일하게 확인됨, 2026-11-22)
EXCLUDED_CODES = {"004730", "217590"}

# 거래일 근사치 (1개월≈21거래일, 3개월≈63거래일, 12개월≈252거래일)
TD_1M, TD_3M, TD_12M = 21, 63, 252

MACRO_COLS = ["base_rate", "fx_change", "term_spread", "cpi_yoy", "kospi_rv"]

FEATURE_GROUPS = {
    "return": ["ret_1m", "ret_3m", "mom_12_2", "vol_12m"],
    "size": ["log_mktcap", "log_price", "share_issuance_12m"],
    "liquidity": ["turnover_1m", "abnormal_turnover", "amihud_illiq"],
    "individual_flow": ["individual_net_intensity", "individual_trade_intensity"],
    "foreign_flow": ["foreign_net_intensity", "foreign_trade_intensity"],
    "foreign_holding": [
        "foreign_own_ratio", "foreign_own_chg_1m", "foreign_own_chg_3m", "foreign_limit_exhaustion",
    ],
    "macro": MACRO_COLS,
}
ALL_FEATURES = [f for cols in FEATURE_GROUPS.values() for f in cols]
FEATURE_TO_GROUP = {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}
TARGET = "target_next"

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    missing=np.nan,
)
N_CORES = __import__("os").cpu_count() or 1

# 006 VAL Rank IC (월별 baseline, 006/README.md "결과" 표 그대로)
REFERENCE_MONTHLY = {
    "005 step4 (매매동향+매크로 13개, 월별)": 0.0416,
    "005 step7 (+뉴스 attention, 월별)": 0.0540,
    "006 (23-factor, 월별)": 0.0869,
}


# ===========================================================================
# 공용 함수
# ===========================================================================

def load_panel() -> pd.DataFrame:
    return pd.read_csv(FEATURES_PATH, dtype={"date": str, "code": str})


def select_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    start, end = SPLITS[split]
    return df[(df["date"] >= start) & (df["date"] <= end)]


def daily_ic(dates, pred, actual) -> pd.Series:
    """일별 횡단면 Spearman 상관 (= Rank IC)."""
    tmp = pd.DataFrame({
        "date": np.asarray(dates),
        "pred": np.asarray(pred, dtype=float),
        "actual": np.asarray(actual, dtype=float),
    }).dropna()
    tmp["pred_r"] = tmp.groupby("date")["pred"].rank()
    tmp["actual_r"] = tmp.groupby("date")["actual"].rank()
    # groupby.apply가 pandas 버전에 따라 Series/DataFrame 중 무엇을 돌려줄지 애매해서
    # (단일 스칼라를 돌려주는 경우에도 모양이 바뀔 수 있음), 직접 순회해서 Series로 고정한다.
    corr_by_date = {
        date: g["pred_r"].corr(g["actual_r"]) if len(g) >= 10 else np.nan
        for date, g in tmp.groupby("date")
    }
    ic = pd.Series(corr_by_date)
    return ic.dropna()


def ic_summary(ic: pd.Series) -> dict:
    std = ic.std()
    return {
        "ic_mean": ic.mean(),
        "ic_std": std,
        "ic_ir": ic.mean() / std if std else np.nan,
        "hit_rate": (ic > 0).mean(),
        "n_days": len(ic),
    }


def train_xgb(train_df: pd.DataFrame, features: list[str], n_jobs: int = N_CORES) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=n_jobs)
    model.fit(train_df[features], train_df[TARGET])
    return model


def evaluate_model(model, df: pd.DataFrame, features: list[str]) -> tuple[dict, pd.Series]:
    pred = model.predict(df[features])
    err = pred - df[TARGET].to_numpy()
    ic = daily_ic(df["date"], pred, df[TARGET])
    metrics = {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        **ic_summary(ic),
    }
    return metrics, ic


def format_metrics(m: dict) -> str:
    return (f"RMSE={m['rmse']:.6f}  MAE={m['mae']:.6f}  Rank IC={m['ic_mean']:.4f}  "
            f"IC-IR={m['ic_ir']:.4f}  양의 IC 비율={m['hit_rate']:.1%} ({m['n_days']}일)")


def load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(n_jobs=N_CORES)
    model.load_model(MODEL_PATH)
    return model


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ===========================================================================
# 단계 1: feature 패널 생성
# ===========================================================================

def _load_tickers() -> list[str]:
    header_df = pd.read_csv(
        RAW / "returns" / "train_2001_2021__adjusted_monthly_return.csv", nrows=0, dtype=str
    )
    codes = [c for c in header_df.columns if c != "month"]
    return sorted(c for c in codes if c not in EXCLUDED_CODES)


def _load_ticker_wide(folder: str, cols: list[str], tickers: list[str]) -> dict[str, pd.DataFrame]:
    """folder(raw/ 하위, 파일 하나=종목 하나)에서 cols를 읽어 {col: 날짜×종목 wide} 로 반환."""
    per_ticker = {}
    for t in tqdm(tickers, desc=f"{folder} 로드"):
        path = RAW / folder / f"{t}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path, dtype={"티커": str})
        d = d.drop_duplicates("날짜").set_index("날짜")
        per_ticker[t] = d[cols].apply(pd.to_numeric, errors="coerce")

    all_dates = sorted(set().union(*(d.index for d in per_ticker.values())))
    return {
        c: pd.DataFrame({t: d[c] for t, d in per_ticker.items()}).reindex(index=all_dates, columns=tickers)
        for c in cols
    }, all_dates


def _pos(df: pd.DataFrame) -> pd.DataFrame:
    return df.where(df > 0)


def _ecos_series(name: str) -> pd.Series:
    """ECOS raw/daily_macro/{name}.csv: TIME은 'YYYYMMDD'(구분자 없음) -> 'YYYY-MM-DD'로 변환."""
    d = pd.read_csv(RAW / "daily_macro" / f"{name}.csv", dtype={"TIME": str})
    d["date"] = d["TIME"].str[:4] + "-" + d["TIME"].str[4:6] + "-" + d["TIME"].str[6:8]
    return d.drop_duplicates("date").set_index("date")["DATA_VALUE"]


def _load_macro_daily(dates: list[str]) -> pd.DataFrame:
    """raw/daily_macro(환율·금리) + raw/macro(cpi_yoy만, 월 단위 그대로 그 달 모든 날에 전파)."""
    fx = _ecos_series("fx_usd_krw")
    t3 = _ecos_series("treasury_3y")
    t10 = _ecos_series("treasury_10y")
    base = _ecos_series("base_rate")

    idx = pd.Index(dates, name="date")
    fx = pd.to_numeric(fx, errors="coerce").reindex(idx)
    t3 = pd.to_numeric(t3, errors="coerce").reindex(idx)
    t10 = pd.to_numeric(t10, errors="coerce").reindex(idx)
    base = pd.to_numeric(base, errors="coerce").reindex(idx)

    out = pd.DataFrame(index=idx)
    out["base_rate"] = base
    out["fx_change"] = fx.pct_change(fill_method=None)
    out["term_spread"] = t10 - t3

    monthly = pd.concat(
        [pd.read_csv(RAW / "macro" / p, dtype={"ym": str}) for p in ("train.csv", "test.csv")],
        ignore_index=True,
    )
    monthly["ym6"] = monthly["ym"]
    cpi = monthly.set_index("ym6")["cpi_yoy"]
    out["cpi_yoy"] = pd.Series(idx.str[:4] + idx.str[5:7], index=idx).map(cpi)
    out["kospi_rv"] = np.nan  # _build_wide_features에서 전종목 평균 일별수익률로 대체 계산
    return out


def _build_wide_features():
    tickers = _load_tickers()

    price_wide, _ = _load_ticker_wide("daily_price_adjusted", ["수정종가", "거래량"], tickers)
    adj_close = price_wide["수정종가"]
    volume = price_wide["거래량"]
    dates = list(adj_close.index)
    print(f"[daily_price_adjusted] {dates[0]} ~ {dates[-1]} ({len(dates)}거래일), 종목 {len(tickers)}개")

    raw_price_wide, _ = _load_ticker_wide("daily_price", ["종가"], tickers)
    raw_close = raw_price_wide["종가"].reindex(index=dates, columns=tickers)

    mcap_wide, _ = _load_ticker_wide("daily_marketcap", ["시가총액", "상장주식수", "거래대금"], tickers)
    mcap = mcap_wide["시가총액"].reindex(index=dates, columns=tickers)
    shares = mcap_wide["상장주식수"].reindex(index=dates, columns=tickers)
    value = mcap_wide["거래대금"].reindex(index=dates, columns=tickers)

    inv_wide, _ = _load_ticker_wide(
        "daily_investor_flow",
        ["개인_매수거래대금", "개인_매도거래대금", "외국인합계_매수거래대금", "외국인합계_매도거래대금"],
        tickers,
    )
    ind_buy = inv_wide["개인_매수거래대금"].reindex(index=dates, columns=tickers)
    ind_sell = inv_wide["개인_매도거래대금"].reindex(index=dates, columns=tickers)
    frn_buy = inv_wide["외국인합계_매수거래대금"].reindex(index=dates, columns=tickers)
    frn_sell = inv_wide["외국인합계_매도거래대금"].reindex(index=dates, columns=tickers)

    fh_wide, _ = _load_ticker_wide(
        "daily_foreign_holding", ["상장주식수", "보유수량", "한도소진률"], tickers
    )
    fh_shares = fh_wide["상장주식수"].reindex(index=dates, columns=tickers)
    fh_held = fh_wide["보유수량"].reindex(index=dates, columns=tickers)
    fh_limit = fh_wide["한도소진률"].reindex(index=dates, columns=tickers)

    f = {}

    # --- return
    # 거래정지(거래량=0)로 멈춰있던 가격이 재개되며 수백만%짜리 가짜 수익률이 생기는 걸
    # 막기 위해: (1) 거래정지 날 전후 수익률, (2) KRX 가격제한폭(±30%)을 넘는 수익률을
    # 결측 처리한다(2026-11-22, 008080 2013-09-11 +6,699,900%로 발견).
    daily_ret = adj_close.pct_change(fill_method=None)
    halted = (volume == 0) | (volume.shift(1) == 0)
    daily_ret = daily_ret.where(~halted & (daily_ret.abs() <= 0.30))
    log_ret = np.log1p(daily_ret)
    f["ret_1m"] = np.exp(log_ret.rolling(TD_1M, min_periods=TD_1M).sum()) - 1
    f["ret_3m"] = np.exp(log_ret.rolling(TD_3M, min_periods=TD_3M).sum()) - 1
    f["mom_12_2"] = np.exp(
        log_ret.shift(TD_1M).rolling(TD_12M - TD_1M, min_periods=TD_12M - TD_1M).sum()
    ) - 1
    f["vol_12m"] = daily_ret.rolling(TD_12M, min_periods=TD_3M).std()

    # --- size (log_price는 006과 동일하게 raw 종가 사용, 수정종가 아님)
    f["log_mktcap"] = np.log(_pos(mcap))
    f["log_price"] = np.log(_pos(raw_close))
    f["share_issuance_12m"] = np.log(_pos(shares) / _pos(shares.shift(TD_12M)))

    # --- liquidity
    f["turnover_1m"] = value / _pos(mcap)
    past_turnover = f["turnover_1m"].shift(TD_1M).rolling(TD_12M, min_periods=TD_3M).mean()
    f["abnormal_turnover"] = f["turnover_1m"] / _pos(past_turnover)
    f["amihud_illiq"] = daily_ret.abs() / _pos(value / 1e8)

    # --- individual_flow / foreign_flow
    f["individual_net_intensity"] = (ind_buy - ind_sell) / _pos(mcap)
    f["individual_trade_intensity"] = (ind_buy + ind_sell) / _pos(mcap)
    f["foreign_net_intensity"] = (frn_buy - frn_sell) / _pos(mcap)
    f["foreign_trade_intensity"] = (frn_buy + frn_sell) / _pos(mcap)

    # --- foreign_holding (PREPROCESSING.md 8번: 비정상 비율은 결측 처리, raw 원본은 안 건드림)
    own = fh_held / _pos(fh_shares)
    own = own.where((own >= 0) & (own <= 2))
    limit = fh_limit.where((fh_limit >= 0) & (fh_limit <= 200))
    f["foreign_own_ratio"] = own
    f["foreign_own_chg_1m"] = own - own.shift(TD_1M)
    f["foreign_own_chg_3m"] = own - own.shift(TD_3M)
    f["foreign_limit_exhaustion"] = limit

    # --- macro (일별 환율/금리 + 월별 cpi_yoy 전파), kospi_rv는 전종목 평균 일별수익률 표준편차로 근사
    macro = _load_macro_daily(dates)
    macro["kospi_rv"] = daily_ret.mean(axis=1).rolling(TD_1M, min_periods=TD_1M).std()
    for col in MACRO_COLS:
        f[col] = pd.DataFrame(
            np.tile(macro[col].to_numpy().reshape(-1, 1), (1, len(tickers))),
            index=dates, columns=tickers,
        )

    f[TARGET] = daily_ret.shift(-1)  # 다음 거래일 수익률 (거래정지·극단치 결측 처리된 값)
    return f, mcap, dates, tickers


def _to_long(wide: dict, mcap: pd.DataFrame, dates: list[str], tickers: list[str]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "code"])
    data = {name: w.to_numpy().ravel() for name, w in wide.items()}
    data["_mcap"] = mcap.to_numpy().ravel()
    long = pd.DataFrame(data, index=idx).reset_index()
    long = long.replace([np.inf, -np.inf], np.nan)

    long = long[(long["date"] >= OUTPUT_START) & (long["date"] <= OUTPUT_END)]
    long = long[long["_mcap"].notna()]
    long = long[long[TARGET].notna()]
    return long.drop(columns="_mcap")


def stage_features(rebuild: bool) -> None:
    header("[features] feature 패널 생성 (일별)")
    if FEATURES_PATH.exists() and not rebuild:
        print(f"이미 존재 -> {FEATURES_PATH} (다시 만들려면 --rebuild)")
        return

    wide, mcap, dates, tickers = _build_wide_features()
    panel = _to_long(wide, mcap, dates, tickers)
    panel = panel[["date", "code"] + ALL_FEATURES + [TARGET]]

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(FEATURES_PATH, index=False)
    print(f"\n저장 완료 -> {FEATURES_PATH}, 행수={len(panel)}")

    for split in ("train", "val"):
        part = select_split(panel, split)
        print(f"[{split}] 행수={len(part)}, 거래일 수={part['date'].nunique()}, "
              f"거래일당 평균 종목 수={len(part) / max(part['date'].nunique(), 1):.0f}")
    print("\n--- train 구간 feature 커버리지 (결측 아닌 비율) ---")
    cov = select_split(panel, "train")[ALL_FEATURES].notna().mean()
    for name, v in cov.items():
        print(f"{name:30s} {v:6.1%}")


# ===========================================================================
# 단계 2: 단일 팩터 Rank IC
# ===========================================================================

def stage_factor_ic(df: pd.DataFrame) -> None:
    header("[factor_ic] 단일 팩터 Rank IC (feature vs 다음 거래일 수익률)")
    features = [f for f in ALL_FEATURES if f not in MACRO_COLS]

    rows = []
    for feat in tqdm(features, desc="factor IC"):
        row = {"feature": feat, "group": FEATURE_TO_GROUP[feat]}
        for split in ("train", "val"):
            part = select_split(df, split)
            s = ic_summary(daily_ic(part["date"], part[feat], part[TARGET]))
            row[f"{split}_coverage"] = part[feat].notna().mean()
            row[f"{split}_ic"] = s["ic_mean"]
            row[f"{split}_ic_ir"] = s["ic_ir"]
            row[f"{split}_hit"] = s["hit_rate"]
        row["sign_consistent"] = np.sign(row["train_ic"]) == np.sign(row["val_ic"])
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("train_ic", key=lambda s: -s.abs())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS_DIR / "factor_ic.csv", index=False)

    print(result[[
        "feature", "group", "train_ic", "train_ic_ir", "train_hit",
        "val_ic", "val_ic_ir", "val_hit", "sign_consistent",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot_df = result.sort_values("train_ic")
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(plot_df) + 1.5))
    ax.barh(y - 0.2, plot_df["train_ic"], height=0.4, label="train 2001-2017")
    ax.barh(y + 0.2, plot_df["val_ic"], height=0.4, label="val 2018-2021")
    ax.set_yticks(y, plot_df["feature"])
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("mean daily Rank IC (feature vs next-day return)")
    ax.legend(loc="lower right")
    ax.set_title("Single-factor Rank IC (daily)")
    fig.tight_layout()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES_DIR / "factor_ic.png", dpi=150)
    plt.close(fig)
    print(f"\n저장 완료 -> {RESULTS_DIR / 'factor_ic.csv'}, {IMAGES_DIR / 'factor_ic.png'}")


# ===========================================================================
# 단계 3: 전체 feature 학습 / 평가
# ===========================================================================

def stage_train(df: pd.DataFrame) -> None:
    header("[train] 전체 feature XGBoost 학습")
    train_df = select_split(df, "train")
    print(f"train 행수={len(train_df)}, feature {len(ALL_FEATURES)}개")
    model = train_xgb(train_df, ALL_FEATURES)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"모델 저장 완료 -> {MODEL_PATH}")


def stage_evaluate(df: pd.DataFrame, split: str) -> None:
    header(f"[evaluate] 전체 feature 모델 평가 ({split})")
    split_df = select_split(df, split)
    metrics, ic = evaluate_model(load_model(), split_df, ALL_FEATURES)
    print(f"[{split}] 행수={len(split_df)}")
    print(f"[{split}] {format_metrics(metrics)}")

    if split == "val":
        print("\n참고 — 006/005 VAL Rank IC (월별):")
        for name, v in REFERENCE_MONTHLY.items():
            print(f"  {name:40s} {v:.4f}")

    print("\n--- 연도별 Rank IC ---")
    for year, s in ic.groupby(ic.index.str[:4]):
        y = ic_summary(s)
        print(f"{year}  IC={y['ic_mean']:.4f}  IC-IR={y['ic_ir']:.4f}  양의 IC 비율={y['hit_rate']:.1%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"daily_ic_{split}.csv"
    pd.DataFrame({"date": ic.index, "ic": ic.to_numpy()}).to_csv(out, index=False)
    print(f"\n일별 IC 저장 완료 -> {out}")


# ===========================================================================
# main
# ===========================================================================

STAGES = ["features", "factor_ic", "train", "evaluate"]


def main() -> None:
    parser = argparse.ArgumentParser(description="015 일별 팩터 패널 실험")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES,
                        help="실행할 단계 (기본: 전체)")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="evaluate 대상 구간 (test는 최종 1회만)")
    parser.add_argument("--rebuild", action="store_true", help="feature 패널을 다시 생성")
    args = parser.parse_args()

    if args.split == "test":
        print("주의: test는 최종 평가 전용입니다. 모델/feature가 다 정해진 뒤에만 실행하세요.")

    if "features" in args.stages:
        stage_features(args.rebuild)

    rest = [s for s in args.stages if s != "features"]
    if not rest:
        return
    df = load_panel()

    for stage in STAGES:
        if stage not in rest:
            continue
        if stage == "factor_ic":
            stage_factor_ic(df)
        elif stage == "train":
            stage_train(df)
        elif stage == "evaluate":
            stage_evaluate(df, args.split)


if __name__ == "__main__":
    main()
