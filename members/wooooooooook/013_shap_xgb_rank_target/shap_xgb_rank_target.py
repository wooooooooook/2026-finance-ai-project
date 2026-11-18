"""
shap_xgb_rank_target.py: 013 실험 — 007의 9개 모델 중 CV IC 1위(0.1057)였던 xgb_rank_target을
train 분할로 다시 학습시켜 SHAP을 뽑는다. 007은 모델을 저장하지 않아서 다시 학습이 필요한데,
007/model_comparison.py와 같은 feature·하이퍼파라미터·seed를 이 폴더 안에 독립적으로 복사해서
같은 모델을 재현한다(VAL Rank IC가 007 로그의 0.1008과 같은지로 확인 — 아래 출력 첫 줄).
뉴스를 전혀 안 써서 FinBERT 임베딩 단계가 없고, 재무·거래 feature 패널 생성만 하면 되므로
008/009/011/012보다 훨씬 빠르게 끝난다.

SHAP은 006(xgboost_factors.py)과 같은 방식: xgboost 내장 TreeSHAP(pred_contribs)이
shap.TreeExplainer와 같은 값을 멀티코어로 계산해준다.

실행 순서:
  1. 종목x월 재무·거래 feature 패널 (006/007/008/009와 동일). features/panel.csv.gz가
     있으면 재사용
  2. 단일 팩터 Rank IC (006과 동일 — 모델 없이 feature 하나씩, train/val 부호 일관성).
     모델이 SHAP에서 그 feature를 실제로 어떻게 쓰는지와 비교하는 기준선
  3. xgb_rank_target을 train(2001-2017)으로 학습 → val(2018-2021) Rank IC로 재현 확인
  4. VAL에서 SHAP 계산 → feature별/그룹별 중요도 + beeswarm/그룹 막대/dependence 이미지

사용법:
    cd members/wooooooooook/013_shap_xgb_rank_target
    python3 shap_xgb_rank_target.py
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from tqdm import tqdm

# ===========================================================================
# 설정 (006/007/008/009와 동일)
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
RAW = PROJECT_ROOT / "raw"

FEATURES_PATH = BASE_DIR / "features" / "panel.csv.gz"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"

SPLITS = {
    "train": ("2001-01", "2017-12"),
    "val": ("2018-01", "2021-12"),
    "test": ("2022-01", "2025-12"),
}
RETURN_PERIODS = ["warmup_2000", "train_2001_2021", "test_2022_2025"]
OUTPUT_START, OUTPUT_END = "2001-01", "2025-12"
EXCLUDED_CODES = {"004730", "217590"}

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
TARGET_RANK = "target_rank"

N_CORES = os.cpu_count() or 1
SEED = 42
N_TREES, DEPTH, LR = 500, 4, 0.05

SHAP_MAX_ROWS = 20000
SHAP_TOP_K = 6


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ===========================================================================
# 1. 재무·거래 feature 패널 (006/007/008/009와 동일 — 독립 복사)
# ===========================================================================

def _load_returns_wide(var: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(RAW / "returns" / f"{p}__{var}.csv", dtype={"month": str}).set_index("month")
        for p in RETURN_PERIODS
    ]
    df = pd.concat(frames).sort_index().astype(float)
    df.columns = df.columns.astype(str)
    return df


def _read_krx_month(folder: str, cols: list[str], month: str) -> pd.DataFrame:
    path = RAW / "trading_flow" / folder / f"{month.replace('-', '')}.csv"
    d = pd.read_csv(path, dtype={"티커": str}, encoding="utf-8-sig")
    d = d.drop_duplicates("티커").set_index("티커")
    return d[cols].apply(pd.to_numeric, errors="coerce")


def _load_krx_wide(folder: str, cols: list[str], months: list[str], codes: list[str]) -> dict:
    with ThreadPoolExecutor(max_workers=N_CORES) as pool:
        frames = list(tqdm(
            pool.map(lambda m: _read_krx_month(folder, cols, m), months),
            total=len(months), desc=f"krx {folder}",
        ))
    per_col = {c: {m: d[c] for m, d in zip(months, frames)} for c in cols}
    return {
        c: pd.DataFrame(per_col[c]).T.reindex(index=months, columns=codes).astype(float)
        for c in cols
    }


def _pos(df: pd.DataFrame) -> pd.DataFrame:
    return df.where(df > 0)


def stage_features() -> None:
    header("[1] 재무·거래 feature 패널")
    if FEATURES_PATH.exists():
        print(f"이미 존재 -> {FEATURES_PATH}")
        return

    ret = _load_returns_wide("adjusted_monthly_return")
    months, codes = list(ret.index), list(ret.columns)
    idx = pd.PeriodIndex(months, freq="M")
    assert (idx == pd.period_range(idx[0], idx[-1], freq="M")).all(), "월 인덱스가 연속이 아님"
    mcap, price, shares = (
        _load_returns_wide(v).reindex(index=months, columns=codes)
        for v in ("raw_month_end_market_cap", "raw_month_end_price", "raw_month_end_shares_outstanding")
    )
    pc = _load_krx_wide("price_change", ["거래대금"], months, codes)
    ind = _load_krx_wide("net_individual", ["매도거래대금", "매수거래대금"], months, codes)
    frn = _load_krx_wide("net_foreign", ["매도거래대금", "매수거래대금"], months, codes)
    fh = _load_krx_wide("foreign_holding", ["상장주식수", "보유수량", "한도소진률"], months, codes)

    f = {}
    log_gross = np.log(_pos(1 + ret))
    f["ret_1m"] = ret
    f["ret_3m"] = np.exp(log_gross.rolling(3, min_periods=3).sum()) - 1
    f["mom_12_2"] = np.exp(log_gross.shift(1).rolling(11, min_periods=11).sum()) - 1
    f["vol_12m"] = ret.rolling(12, min_periods=6).std()
    f["log_mktcap"] = np.log(_pos(mcap))
    f["log_price"] = np.log(_pos(price))
    f["share_issuance_12m"] = np.log(_pos(shares) / _pos(shares.shift(12)))
    value = pc["거래대금"]
    f["turnover_1m"] = value / _pos(mcap)
    past_turnover = f["turnover_1m"].shift(1).rolling(12, min_periods=6).mean()
    f["abnormal_turnover"] = f["turnover_1m"] / _pos(past_turnover)
    f["amihud_illiq"] = ret.abs() / _pos(value / 1e8)
    f["individual_net_intensity"] = (ind["매수거래대금"] - ind["매도거래대금"]) / _pos(mcap)
    f["individual_trade_intensity"] = (ind["매수거래대금"] + ind["매도거래대금"]) / _pos(mcap)
    f["foreign_net_intensity"] = (frn["매수거래대금"] - frn["매도거래대금"]) / _pos(mcap)
    f["foreign_trade_intensity"] = (frn["매수거래대금"] + frn["매도거래대금"]) / _pos(mcap)
    own = fh["보유수량"] / _pos(fh["상장주식수"])
    f["foreign_own_ratio"] = own
    f["foreign_own_chg_1m"] = own - own.shift(1)
    f["foreign_own_chg_3m"] = own - own.shift(3)
    f["foreign_limit_exhaustion"] = fh["한도소진률"]
    f[TARGET] = ret.shift(-1)

    mi = pd.MultiIndex.from_product([months, codes], names=["month", "code"])
    data = {name: w.to_numpy().ravel() for name, w in f.items()}
    data["_mcap"] = mcap.to_numpy().ravel()
    long = pd.DataFrame(data, index=mi).reset_index().replace([np.inf, -np.inf], np.nan)
    long = long[(long["month"] >= OUTPUT_START) & (long["month"] <= OUTPUT_END)]
    long = long[~long["code"].isin(EXCLUDED_CODES) & long["_mcap"].notna() & long[TARGET].notna()]

    macro = pd.concat(
        [pd.read_csv(RAW / "macro" / p, dtype={"ym": str}) for p in ("train.csv", "test.csv")],
        ignore_index=True,
    )
    macro["month"] = macro["ym"].str[:4] + "-" + macro["ym"].str[4:]
    panel = long.merge(macro[["month"] + MACRO_COLS], on="month", how="left")
    panel = panel[["month", "code"] + ALL_FEATURES + [TARGET]]

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(FEATURES_PATH, index=False)
    print(f"저장 완료 -> {FEATURES_PATH}, 행수={len(panel)}")


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, dtype={"month": str, "code": str})
    df[TARGET_RANK] = df.groupby("month")[TARGET].rank(pct=True) - 0.5
    return df


def select_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    start, end = SPLITS[split]
    return df[(df["month"] >= start) & (df["month"] <= end)]


def monthly_ic(months, pred, actual) -> pd.Series:
    tmp = pd.DataFrame({
        "month": np.asarray(months),
        "pred": np.asarray(pred, dtype=float),
        "actual": np.asarray(actual, dtype=float),
    }).dropna()
    tmp["pred_r"] = tmp.groupby("month")["pred"].rank()
    tmp["actual_r"] = tmp.groupby("month")["actual"].rank()
    ic = tmp.groupby("month")[["pred_r", "actual_r"]].apply(
        lambda g: g["pred_r"].corr(g["actual_r"]) if len(g) >= 10 else np.nan
    )
    return ic.dropna()


def ic_summary(ic: pd.Series) -> dict:
    std = ic.std()
    return {
        "ic_mean": ic.mean(),
        "ic_ir": ic.mean() / std if std else np.nan,
        "hit_rate": (ic > 0).mean(),
        "n_months": len(ic),
    }


# ===========================================================================
# 2. 단일 팩터 Rank IC (006과 동일 — 모델 없이 feature 하나씩, 모델이 그 feature를
#    실제로 어떻게 쓰는지와 비교하는 기준선. 매크로는 월 단위 상수라 단일 팩터 IC가
#    구조적으로 의미 없어서(006과 동일하게) 제외)
# ===========================================================================

def stage_factor_ic(df: pd.DataFrame) -> None:
    header("[2] 단일 팩터 Rank IC (feature vs t+1월 수익률)")
    features = [f for f in ALL_FEATURES if f not in MACRO_COLS]

    rows = []
    for feat in tqdm(features, desc="factor IC"):
        row = {"feature": feat, "group": FEATURE_TO_GROUP[feat]}
        for split in ("train", "val"):
            part = select_split(df, split)
            s = ic_summary(monthly_ic(part["month"], part[feat], part[TARGET]))
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
    ax.set_xlabel("mean monthly Rank IC (feature vs next-month return)")
    ax.legend(loc="lower right")
    ax.set_title("Single-factor Rank IC (xgb_rank_target feature set)")
    fig.tight_layout()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES_DIR / "factor_ic.png", dpi=150)
    plt.close(fig)
    print(f"\n저장 완료 -> {RESULTS_DIR / 'factor_ic.csv'}, {IMAGES_DIR / 'factor_ic.png'}")


# ===========================================================================
# 3. xgb_rank_target 재학습 + VAL IC로 재현 확인
# ===========================================================================

def stage_train_and_check(df: pd.DataFrame) -> tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame]:
    header("[3] xgb_rank_target 재학습 (007과 동일 설정)")
    train_df, val_df = select_split(df, "train"), select_split(df, "val")
    print(f"train={len(train_df)}, val={len(val_df)}")

    model = xgb.XGBRegressor(
        n_estimators=N_TREES, max_depth=DEPTH, learning_rate=LR,
        random_state=SEED, n_jobs=N_CORES, missing=np.nan,
    )
    model.fit(train_df[ALL_FEATURES], train_df[TARGET_RANK].to_numpy())

    pred = model.predict(val_df[ALL_FEATURES])
    ic = monthly_ic(val_df["month"], pred, val_df[TARGET])
    print(f"검증: VAL Rank IC={ic.mean():.4f} (007 로그의 xgb_rank_target=0.1008과 같아야 함)")
    return model, train_df, val_df


# ===========================================================================
# 4. SHAP (006과 동일 방식: xgboost 내장 TreeSHAP)
# ===========================================================================

def stage_shap(model: xgb.XGBRegressor, val_df: pd.DataFrame) -> None:
    header("[4] SHAP 중요도 (VAL)")
    X = val_df[ALL_FEATURES]
    if len(X) > SHAP_MAX_ROWS:
        X = X.sample(SHAP_MAX_ROWS, random_state=SEED)
    print(f"SHAP 분석 대상 행수: {len(X)}")

    booster = model.get_booster()
    booster.set_param({"nthread": N_CORES})
    contribs = booster.predict(xgb.DMatrix(X, missing=np.nan), pred_contribs=True)
    shap_values = shap.Explanation(
        values=contribs[:, :-1], base_values=contribs[:, -1],
        data=X.to_numpy(), feature_names=ALL_FEATURES,
    )

    imp = pd.DataFrame({
        "feature": ALL_FEATURES,
        "group": [FEATURE_TO_GROUP[f] for f in ALL_FEATURES],
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    group_imp = imp.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)

    print("\n--- feature별 평균 |SHAP| ---")
    print(imp.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n--- 그룹별 평균 |SHAP| 합 ---")
    for g, v in group_imp.items():
        print(f"{g:20s} {v:.6f}  ({v / group_imp.sum():.1%})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    imp.to_csv(RESULTS_DIR / "shap_importance_val.csv", index=False)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X, show=False, max_display=len(ALL_FEATURES))
    plt.savefig(IMAGES_DIR / "shap_beeswarm_val.png", bbox_inches="tight", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    group_imp.sort_values().plot.barh(ax=ax)
    ax.set_xlabel("sum of mean |SHAP| within group")
    ax.set_title("xgb_rank_target: group-level SHAP contribution (VAL)")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "shap_group_val.png", dpi=150)
    plt.close(fig)

    top = imp["feature"].head(SHAP_TOP_K).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feat in zip(axes.ravel(), top):
        x = X[feat].to_numpy()
        y = shap_values.values[:, ALL_FEATURES.index(feat)]
        lo, hi = np.nanpercentile(x, [1, 99])
        mask = ~np.isnan(x) & (x >= lo) & (x <= hi)
        ax.scatter(x[mask], y[mask], s=2, alpha=0.2)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(feat)
        ax.set_xlabel("feature value (1-99 pct)")
        ax.set_ylabel("SHAP")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "shap_dependence_val.png", dpi=150)
    plt.close(fig)

    print(f"\n저장 완료 -> {IMAGES_DIR}/shap_beeswarm_val.png, shap_group_val.png, shap_dependence_val.png")


def main() -> None:
    stage_features()
    df = load_panel()
    stage_factor_ic(df)
    model, _, val_df = stage_train_and_check(df)
    stage_shap(model, val_df)


if __name__ == "__main__":
    main()
