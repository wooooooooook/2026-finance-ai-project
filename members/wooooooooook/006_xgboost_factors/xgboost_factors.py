"""
xgboost_factors.py: 006 실험 — 뉴스 없이 재무·거래 데이터만으로 XGBoost를 학습하고, 각 요소가
다음달 수익률 순위 예측에 어떻게 영향을 미치는지 분석한다. (실험 내용은 README.md 참고)

단계 (--stages로 골라서 실행, 기본은 전체):
  features   원본 → 종목×월 feature 패널 (features/panel.csv.gz). 이미 있으면 재사용 (--rebuild로 강제)
  factor_ic  모델 없이 feature 하나씩 다음달 수익률과의 월별 Rank IC (train/val)
  train      전체 feature로 XGBoost 학습 → models/xgb_all.json
  evaluate   저장된 모델 평가 (전체 지표 + 연도별 IC, 005와 비교)
  shap       SHAP feature별/그룹별 중요도, beeswarm, dependence plot
  ablation   feature 그룹을 빼거나(drop) 그 그룹만 써서(only) 재학습 → VAL IC 변화

test는 --split test로 명시했을 때만 evaluate/shap에서 쓰인다 (최종 1회).

사용법:
    cd members/wooooooooook/006_xgboost_factors
    python3 xgboost_factors.py                                # 전체 단계
    python3 xgboost_factors.py --stages features factor_ic     # 일부 단계만
    python3 xgboost_factors.py --stages evaluate shap --split test  # 최종 test 평가 (모든 결정이 끝난 뒤에만)
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
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

# 005와 동일한 분할 (비교 가능하도록 고정)
SPLITS = {
    "train": ("2001-01", "2017-12"),
    "val": ("2018-01", "2021-12"),
    "test": ("2022-01", "2025-12"),
}
RETURN_PERIODS = ["warmup_2000", "train_2001_2021", "test_2022_2025"]
OUTPUT_START, OUTPUT_END = "2001-01", "2025-12"

# PREPROCESSING.md 7번: 사실상 전 기간 결측인 종목
EXCLUDED_CODES = {"004730", "217590"}

MACRO_COLS = ["base_rate", "fx_change", "term_spread", "cpi_yoy", "kospi_rv"]

# feature 그룹 — ablation, 그룹별 SHAP 합산의 단위
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

# 005 step4와 동일한 하이퍼파라미터 (feature 효과만 비교하기 위해 고정)
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    missing=np.nan,
)

N_CORES = os.cpu_count() or 1
# ablation은 작은 모델 여러 개라, 모델 하나에 전 코어를 주는 것보다 여러 개를 동시에 학습하는 편이 빠름
# (XGBoost 학습은 GIL을 놓으므로 스레드 병렬로 충분)
ABLATION_WORKERS = 4

# 005 README의 VAL Rank IC (비교 기준)
REFERENCE_005 = {
    "005 step3 (매매동향 8개)": 0.0365,
    "005 step4 (+매크로, 13개)": 0.0416,
    "005 step7 (+뉴스 attention)": 0.0540,
}

SHAP_MAX_ROWS = 50_000   # SHAP 계산 시간 제한용 샘플 수
SHAP_TOP_K = 6           # dependence plot을 그릴 상위 feature 수


# ===========================================================================
# 공용 함수
# ===========================================================================

def load_panel() -> pd.DataFrame:
    # code는 "0120G0"처럼 영문이 섞인 경우가 있어 반드시 str로 고정 (005에서 겪은 merge 버그)
    return pd.read_csv(FEATURES_PATH, dtype={"month": str, "code": str})


def select_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    start, end = SPLITS[split]
    return df[(df["month"] >= start) & (df["month"] <= end)]


def monthly_ic(months, pred, actual) -> pd.Series:
    """월별 횡단면 Spearman 상관 (= Rank IC). 월 내 순위를 매긴 뒤 Pearson 상관을 구한다."""
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
        "ic_std": std,
        "ic_ir": ic.mean() / std if std else np.nan,
        "hit_rate": (ic > 0).mean(),
        "n_months": len(ic),
    }


def train_xgb(train_df: pd.DataFrame, features: list[str], n_jobs: int = N_CORES) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=n_jobs)
    model.fit(train_df[features], train_df[TARGET])
    return model


def evaluate_model(model, df: pd.DataFrame, features: list[str]) -> tuple[dict, pd.Series]:
    pred = model.predict(df[features])
    err = pred - df[TARGET].to_numpy()
    ic = monthly_ic(df["month"], pred, df[TARGET])
    metrics = {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        **ic_summary(ic),
    }
    return metrics, ic


def format_metrics(m: dict) -> str:
    return (f"RMSE={m['rmse']:.6f}  MAE={m['mae']:.6f}  Rank IC={m['ic_mean']:.4f}  "
            f"IC-IR={m['ic_ir']:.4f}  양의 IC 비율={m['hit_rate']:.1%} ({m['n_months']}개월)")


def load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(n_jobs=N_CORES)
    model.load_model(MODEL_PATH)
    return model


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ===========================================================================
# 단계 1: feature 패널 생성
#   모든 feature는 t월 말까지 공개된 정보만 사용, 타겟은 t+1월 수익률.
#   종목 자신의 과거 값만으로 계산(횡단면 정규화·통계 없음) → test 정보 누출 없음.
#   변수마다 (월 × 종목) wide 행렬 → shift/rolling으로 lag → long(month, code)으로 펼침.
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


def _check_consecutive(months: list[str]) -> None:
    # shift(k)가 "k개월 전"을 뜻하려면 월이 빠짐없이 연속이어야 함
    idx = pd.PeriodIndex(months, freq="M")
    expected = pd.period_range(idx[0], idx[-1], freq="M")
    assert len(idx) == len(expected) and (idx == expected).all(), "월 인덱스가 연속이 아님"


def _pos(df: pd.DataFrame) -> pd.DataFrame:
    return df.where(df > 0)


def _build_wide_features():
    ret = _load_returns_wide("adjusted_monthly_return")
    months = list(ret.index)
    codes = list(ret.columns)
    _check_consecutive(months)
    mcap, price, shares = (
        _load_returns_wide(v).reindex(index=months, columns=codes)
        for v in ("raw_month_end_market_cap", "raw_month_end_price", "raw_month_end_shares_outstanding")
    )
    print(f"[returns] {months[0]} ~ {months[-1]} ({len(months)}개월), 종목 {len(codes)}개")

    pc = _load_krx_wide("price_change", ["거래대금"], months, codes)
    ind = _load_krx_wide("net_individual", ["매도거래대금", "매수거래대금"], months, codes)
    frn = _load_krx_wide("net_foreign", ["매도거래대금", "매수거래대금"], months, codes)
    fh = _load_krx_wide("foreign_holding", ["상장주식수", "보유수량", "한도소진률"], months, codes)

    f = {}

    # --- return: 수익률 기반 (t월 수익률 포함 — 005는 t-1월부터 썼음)
    log_gross = np.log(_pos(1 + ret))
    f["ret_1m"] = ret                                                                 # t월
    f["ret_3m"] = np.exp(log_gross.rolling(3, min_periods=3).sum()) - 1               # t-2 ~ t
    f["mom_12_2"] = np.exp(log_gross.shift(1).rolling(11, min_periods=11).sum()) - 1  # t-11 ~ t-1
    f["vol_12m"] = ret.rolling(12, min_periods=6).std()                               # t-11 ~ t

    # --- size: 규모·주가 수준·주식수 변화
    f["log_mktcap"] = np.log(_pos(mcap))
    f["log_price"] = np.log(_pos(price))
    f["share_issuance_12m"] = np.log(_pos(shares) / _pos(shares.shift(12)))

    # --- liquidity: 거래회전율·이상거래량·비유동성
    value = pc["거래대금"]
    f["turnover_1m"] = value / _pos(mcap)
    past_turnover = f["turnover_1m"].shift(1).rolling(12, min_periods=6).mean()
    f["abnormal_turnover"] = f["turnover_1m"] / _pos(past_turnover)
    f["amihud_illiq"] = ret.abs() / _pos(value / 1e8)  # |수익률| / 거래대금(억원)

    # --- individual_flow / foreign_flow: AGENTS.md 정의. 외국인 빈칸은 0으로 채우지 않음
    f["individual_net_intensity"] = (ind["매수거래대금"] - ind["매도거래대금"]) / _pos(mcap)
    f["individual_trade_intensity"] = (ind["매수거래대금"] + ind["매도거래대금"]) / _pos(mcap)
    f["foreign_net_intensity"] = (frn["매수거래대금"] - frn["매도거래대금"]) / _pos(mcap)
    f["foreign_trade_intensity"] = (frn["매수거래대금"] + frn["매도거래대금"]) / _pos(mcap)

    # --- foreign_holding: 지분율은 정밀도 문제로 보유수량/상장주식수로 재계산 (trading_flow README)
    own = fh["보유수량"] / _pos(fh["상장주식수"])
    f["foreign_own_ratio"] = own
    f["foreign_own_chg_1m"] = own - own.shift(1)
    f["foreign_own_chg_3m"] = own - own.shift(3)
    f["foreign_limit_exhaustion"] = fh["한도소진률"]

    f[TARGET] = ret.shift(-1)
    return f, mcap, months, codes


def _to_long(wide: dict, mcap: pd.DataFrame, months: list[str], codes: list[str]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([months, codes], names=["month", "code"])
    data = {name: w.to_numpy().ravel() for name, w in wide.items()}
    data["_mcap"] = mcap.to_numpy().ravel()
    long = pd.DataFrame(data, index=idx).reset_index()
    long = long.replace([np.inf, -np.inf], np.nan)

    long = long[(long["month"] >= OUTPUT_START) & (long["month"] <= OUTPUT_END)]
    long = long[~long["code"].isin(EXCLUDED_CODES)]
    long = long[long["_mcap"].notna()]      # t월에 상장·거래된 종목만
    long = long[long[TARGET].notna()]       # 타겟 없는 행 제외 (2025-12, 상장폐지 직전월 등)
    return long.drop(columns="_mcap")


def _attach_macro(long: pd.DataFrame) -> pd.DataFrame:
    macro = pd.concat(
        [pd.read_csv(RAW / "macro" / p, dtype={"ym": str}) for p in ("train.csv", "test.csv")],
        ignore_index=True,
    )
    macro["month"] = macro["ym"].str[:4] + "-" + macro["ym"].str[4:]
    return long.merge(macro[["month"] + MACRO_COLS], on="month", how="left")


def stage_features(rebuild: bool) -> None:
    header("[features] feature 패널 생성")
    if FEATURES_PATH.exists() and not rebuild:
        print(f"이미 존재 -> {FEATURES_PATH} (다시 만들려면 --rebuild)")
        return

    wide, mcap, months, codes = _build_wide_features()
    panel = _attach_macro(_to_long(wide, mcap, months, codes))
    panel = panel[["month", "code"] + ALL_FEATURES + [TARGET]]

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(FEATURES_PATH, index=False)
    print(f"\n저장 완료 -> {FEATURES_PATH}, 행수={len(panel)}")

    for split in ("train", "val"):
        part = select_split(panel, split)
        print(f"[{split}] 행수={len(part)}, 월 수={part['month'].nunique()}, "
              f"월평균 종목 수={len(part) / part['month'].nunique():.0f}")
    print("\n--- train 구간 feature 커버리지 (결측 아닌 비율) ---")
    cov = select_split(panel, "train")[ALL_FEATURES].notna().mean()
    for name, v in cov.items():
        print(f"{name:30s} {v:6.1%}")


# ===========================================================================
# 단계 2: 단일 팩터 Rank IC (모델 없음)
#   IC 부호 +: 값이 클수록 다음달 수익률이 높은 경향, -: 반대.
#   train/val 부호가 같으면 시기가 바뀌어도 유지되는 관계라는 근거.
#   매크로는 그 달 전 종목에 같은 값이라 횡단면 순위가 없어 제외.
# ===========================================================================

def stage_factor_ic(df: pd.DataFrame) -> None:
    header("[factor_ic] 단일 팩터 Rank IC (feature vs t+1월 수익률)")
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
    ax.set_title("Single-factor Rank IC")
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
        print("\n참고 — 005 VAL Rank IC:")
        for name, v in REFERENCE_005.items():
            print(f"  {name:30s} {v:.4f}")

    print("\n--- 연도별 Rank IC ---")
    for year, s in ic.groupby(ic.index.str[:4]):
        y = ic_summary(s)
        print(f"{year}  IC={y['ic_mean']:.4f}  IC-IR={y['ic_ir']:.4f}  양의 IC 비율={y['hit_rate']:.1%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"monthly_ic_{split}.csv"
    pd.DataFrame({"month": ic.index, "ic": ic.to_numpy()}).to_csv(out, index=False)
    print(f"\n월별 IC 저장 완료 -> {out}")


# ===========================================================================
# 단계 4: SHAP — 모델이 각 feature를 얼마나·어떤 모양으로 쓰는가
#   (실제 예측력 기여는 ablation으로 확인. SHAP이 커도 기여가 없을 수 있음)
# ===========================================================================

def stage_shap(df: pd.DataFrame, split: str) -> None:
    import shap

    header(f"[shap] SHAP 중요도 ({split})")
    X = select_split(df, split)[ALL_FEATURES]
    if len(X) > SHAP_MAX_ROWS:
        X = X.sample(SHAP_MAX_ROWS, random_state=42)
    print(f"[{split}] SHAP 분석 대상 행수: {len(X)}")

    # XGBoost 내장 TreeSHAP(pred_contribs)은 shap.TreeExplainer와 같은 값을 멀티코어로 계산함
    booster = load_model().get_booster()
    booster.set_param({"nthread": N_CORES})
    contribs = booster.predict(xgb.DMatrix(X, missing=np.nan), pred_contribs=True)
    shap_values = shap.Explanation(
        values=contribs[:, :-1],
        base_values=contribs[:, -1],
        data=X.to_numpy(),
        feature_names=ALL_FEATURES,
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
    imp.to_csv(RESULTS_DIR / f"shap_importance_{split}.csv", index=False)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X, show=False, max_display=len(ALL_FEATURES))
    plt.savefig(IMAGES_DIR / f"shap_beeswarm_{split}.png", bbox_inches="tight", dpi=150)
    plt.close()

    top = imp["feature"].head(SHAP_TOP_K).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feat in zip(axes.ravel(), top):
        x = X[feat].to_numpy()
        y = shap_values.values[:, ALL_FEATURES.index(feat)]
        lo, hi = np.nanpercentile(x, [1, 99])  # 꼬리값이 축을 지배하지 않도록 1~99% 구간만
        mask = ~np.isnan(x) & (x >= lo) & (x <= hi)
        ax.scatter(x[mask], y[mask], s=2, alpha=0.2)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(feat)
        ax.set_xlabel("feature value (1-99 pct)")
        ax.set_ylabel("SHAP")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / f"shap_dependence_{split}.png", dpi=150)
    plt.close(fig)
    print(f"\n그래프 저장 완료 -> {IMAGES_DIR}/shap_beeswarm_{split}.png, shap_dependence_{split}.png")


# ===========================================================================
# 단계 5: 그룹 ablation — 재학습으로 실제 기여도 측정 (VAL만 사용)
#   drop:<그룹> delta가 음수로 클수록 그 그룹이 중요, 양수면 빼는 게 오히려 나음.
#   only:<그룹>은 그 그룹 단독 설명력. 매크로 단독은 횡단면 정보가 없어 제외.
# ===========================================================================

def stage_ablation(df: pd.DataFrame) -> None:
    header("[ablation] feature 그룹 ablation (train 학습 → val 평가)")
    configs = {"full": ALL_FEATURES}
    for g, cols in FEATURE_GROUPS.items():
        configs[f"drop:{g}"] = [f for f in ALL_FEATURES if f not in cols]
    for g, cols in FEATURE_GROUPS.items():
        if g != "macro":
            configs[f"only:{g}"] = cols

    train_df = select_split(df, "train")
    val_df = select_split(df, "val")
    threads_per_model = max(1, N_CORES // ABLATION_WORKERS)
    print(f"동시 학습 {ABLATION_WORKERS}개 × 모델당 {threads_per_model}스레드")

    def run(item):
        name, feats = item
        m, _ = evaluate_model(train_xgb(train_df, feats, n_jobs=threads_per_model), val_df, feats)
        return {"config": name, "n_features": len(feats), **m}

    with ThreadPoolExecutor(max_workers=ABLATION_WORKERS) as pool:
        rows = list(tqdm(pool.map(run, configs.items()), total=len(configs), desc="ablation"))

    result = pd.DataFrame(rows)
    full_ic = result.loc[result["config"] == "full", "ic_mean"].item()
    result["delta_ic_vs_full"] = result["ic_mean"] - full_ic

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS_DIR / "group_ablation.csv", index=False)
    cols = ["config", "n_features", "rmse", "ic_mean", "ic_ir", "hit_rate", "delta_ic_vs_full"]
    print(result[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n저장 완료 -> {RESULTS_DIR / 'group_ablation.csv'}")


# ===========================================================================
# main
# ===========================================================================

STAGES = ["features", "factor_ic", "train", "evaluate", "shap", "ablation"]


def main() -> None:
    parser = argparse.ArgumentParser(description="006 재무 데이터 XGBoost 팩터 실험")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES,
                        help="실행할 단계 (기본: 전체)")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="evaluate/shap 대상 구간 (test는 최종 1회만)")
    parser.add_argument("--rebuild", action="store_true", help="feature 패널을 다시 생성")
    args = parser.parse_args()

    if args.split == "test":
        print("주의: test는 최종 평가 전용입니다. 모델/feature/하이퍼파라미터가 다 정해진 뒤에만 실행하세요.")

    if "features" in args.stages:
        stage_features(args.rebuild)

    rest = [s for s in args.stages if s != "features"]
    if not rest:
        return
    df = load_panel()

    for stage in STAGES:  # 지정 순서와 무관하게 의존성 순서대로 실행
        if stage not in rest:
            continue
        if stage == "factor_ic":
            stage_factor_ic(df)
        elif stage == "train":
            stage_train(df)
        elif stage == "evaluate":
            stage_evaluate(df, args.split)
        elif stage == "shap":
            stage_shap(df, args.split)
        elif stage == "ablation":
            stage_ablation(df)


if __name__ == "__main__":
    main()
