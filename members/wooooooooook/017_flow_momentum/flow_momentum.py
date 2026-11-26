"""
flow_momentum.py: 017 실험 — 007의 재무·거래 feature 23개(base)에 투자자 유형별(개인·외국인)
수급 모멘텀 feature 8개를 더한 버전(flowmom)이 Rank IC를 개선하는지, 같은 모델·같은 분할로 비교한다.

실행 순서:
  1. 종목×월 feature 패널 생성 (007 정의 + 수급 모멘텀 8개). features/panel.csv.gz가 있으면 재사용
  2. train(2001-2017) 학습 → val(2018-2021) 평가, base vs flowmom vs retstyle
  3. walk-forward CV (val 연도 2009~2021, 매 fold 2001-01부터 직전 연도까지 학습), base vs flowmom vs retstyle
  4. 모델별 Δ(flowmom − base) 요약표·연도별 히트맵

사용법:
    uv run python members/wooooooooook/017_flow_momentum/flow_momentum.py
"""

import os
import sys
import time
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
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"

SPLITS = {
    "train": ("2001-01", "2017-12"),
    "val": ("2018-01", "2021-12"),
    "test": ("2022-01", "2025-12"),
}
CV_YEARS = list(range(2009, 2022))
RETURN_PERIODS = ["warmup_2000", "train_2001_2021", "test_2022_2025"]
OUTPUT_START, OUTPUT_END = "2001-01", "2025-12"
EXCLUDED_CODES = {"004730", "217590"}

MACRO_COLS = ["base_rate", "fx_change", "term_spread", "cpi_yoy", "kospi_rv"]
BASE_GROUPS = {
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
FLOW_MOM_COLS = [
    f"{tag}_{name}"
    for tag in ("individual", "foreign")
    for name in ("net_3m", "net_12m", "net_accel", "trade_abnormal")
]
# 가격 모멘텀(ret_3m, mom_12_2)과 같은 방식: 최근 3개월, 직전 1개월을 건너뛴 12-2(t-11~t-1) 평균을 4개 시리즈 모두에 적용
RET_STYLE_COLS = [
    f"rs_{tag}_{series}_{win}"
    for tag in ("individual", "foreign")
    for series in ("net", "trade")
    for win in ("3m", "12_2")
]
VARIANTS = {
    "base": BASE_GROUPS,
    "flowmom": {**BASE_GROUPS, "flow_momentum": FLOW_MOM_COLS},
    "retstyle": {**BASE_GROUPS, "ret_style_flow": RET_STYLE_COLS},
}
PANEL_FEATURES = list(dict.fromkeys(
    f for groups in VARIANTS.values() for cols in groups.values() for f in cols
))
TARGET = "target_next"
TARGET_RANK = "target_rank"

MODEL = "xgb_rank_target"
COMPARE = [v for v in VARIANTS if v != "base"]

N_CORES = os.cpu_count() or 1
N_TREES, DEPTH, LR = 500, 4, 0.05

ALL_FEATURES: list[str] = []
STOCK_FEATURES: list[str] = []


def set_variant(name: str) -> None:
    ALL_FEATURES[:] = [f for cols in VARIANTS[name].values() for f in cols]
    STOCK_FEATURES[:] = [f for f in ALL_FEATURES if f not in MACRO_COLS]


# ===========================================================================
# 공용 함수 (007과 동일)
# ===========================================================================

def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


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


def monthly_long_short(months, pred, actual, q: float = 0.1) -> pd.Series:
    tmp = pd.DataFrame({
        "month": np.asarray(months),
        "pred": np.asarray(pred, dtype=float),
        "actual": np.asarray(actual, dtype=float),
    }).dropna()
    tmp["pct"] = tmp.groupby("month")["pred"].rank(pct=True)
    top = tmp[tmp["pct"] > 1 - q].groupby("month")["actual"].mean()
    bottom = tmp[tmp["pct"] <= q].groupby("month")["actual"].mean()
    return (top - bottom).dropna()


def summarize(ic: pd.Series, ls: pd.Series) -> dict:
    std = ic.std()
    return {
        "ic_mean": ic.mean(),
        "ic_ir": ic.mean() / std if std else np.nan,
        "hit_rate": (ic > 0).mean(),
        "ls_spread": ls.mean(),
        "n_months": len(ic),
    }


# ===========================================================================
# 단계 1: feature 패널 (007 정의 + 수급 모멘텀)
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

    # 수급 모멘텀: 월별 순매수·거래강도의 과거 누적/변화. 외국인 순매수 결측은 채우지 않고 NaN 전파
    for tag in ("individual", "foreign"):
        net, trade = f[f"{tag}_net_intensity"], f[f"{tag}_trade_intensity"]
        f[f"{tag}_net_3m"] = net.rolling(3, min_periods=3).sum()
        f[f"{tag}_net_12m"] = net.rolling(12, min_periods=12).sum()
        f[f"{tag}_net_accel"] = net - net.shift(1).rolling(3, min_periods=3).mean()
        f[f"{tag}_trade_abnormal"] = trade / _pos(trade.shift(1).rolling(12, min_periods=6).mean())

    # 가격 모멘텀 방식: 4개 시리즈에 3m, 12-2 평균. 윈도우의 절반 이상 월이 있으면 있는 월만 평균(0 채우지 않음)
    for tag in ("individual", "foreign"):
        for series in ("net", "trade"):
            x = f[f"{tag}_{series}_intensity"]
            f[f"rs_{tag}_{series}_3m"] = x.rolling(3, min_periods=2).mean()
            f[f"rs_{tag}_{series}_12_2"] = x.shift(1).rolling(11, min_periods=6).mean()

    f[TARGET] = ret.shift(-1)
    return f, mcap, months, codes


def stage_features() -> None:
    header("[features] feature 패널 생성")
    if FEATURES_PATH.exists():
        have = set(pd.read_csv(FEATURES_PATH, nrows=0).columns)
        if set(PANEL_FEATURES) <= have:
            print(f"이미 존재 -> {FEATURES_PATH}")
            return
        print(f"기존 패널에 없는 feature가 있어 다시 생성: {sorted(set(PANEL_FEATURES) - have)}")

    wide, mcap, months, codes = _build_wide_features()
    idx = pd.MultiIndex.from_product([months, codes], names=["month", "code"])
    data = {name: w.to_numpy().ravel() for name, w in wide.items()}
    data["_mcap"] = mcap.to_numpy().ravel()
    long = pd.DataFrame(data, index=idx).reset_index().replace([np.inf, -np.inf], np.nan)
    long = long[(long["month"] >= OUTPUT_START) & (long["month"] <= OUTPUT_END)]
    long = long[~long["code"].isin(EXCLUDED_CODES) & long["_mcap"].notna() & long[TARGET].notna()]

    macro = pd.concat(
        [pd.read_csv(RAW / "macro" / p, dtype={"ym": str}) for p in ("train.csv", "test.csv")],
        ignore_index=True,
    )
    macro["month"] = macro["ym"].str[:4] + "-" + macro["ym"].str[4:]
    panel = long.merge(macro[["month"] + MACRO_COLS], on="month", how="left")
    panel = panel[["month", "code"] + PANEL_FEATURES + [TARGET]]

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(FEATURES_PATH, index=False)
    print(f"저장 완료 -> {FEATURES_PATH}, 행수={len(panel)}")
    print("수급 모멘텀 feature 커버리지(전체 행 대비 비결측):")
    print(panel[FLOW_MOM_COLS + RET_STYLE_COLS].notna().mean().round(3).to_string())


# ===========================================================================
# 모델 (007과 동일한 설정, 사용할 모델만)
# ===========================================================================

class RankScaler:
    def fit(self, df: pd.DataFrame) -> "RankScaler":
        self.mu = df[MACRO_COLS].mean()
        self.sd = df[MACRO_COLS].std().replace(0, 1)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        stock = (df[STOCK_FEATURES].groupby(df["month"]).rank(pct=True) - 0.5).fillna(0.0)
        macro = ((df[MACRO_COLS] - self.mu) / self.sd).fillna(0.0)
        return np.hstack([stock.to_numpy(), macro.to_numpy()])


class Model:
    def __init__(self, est, target: str = TARGET, rank_input: bool = False):
        self.est, self.target, self.rank_input = est, target, rank_input

    def _X(self, df: pd.DataFrame, fit: bool = False):
        if self.rank_input:
            if fit:
                self.scaler = RankScaler().fit(df)
            return self.scaler.transform(df)
        return df[ALL_FEATURES]

    def fit(self, df: pd.DataFrame) -> "Model":
        self.est.fit(self._X(df, fit=True), df[self.target].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.est.predict(self._X(df)), dtype=float)


def make_models() -> dict:
    xgb_kw = dict(n_estimators=N_TREES, max_depth=DEPTH, learning_rate=LR,
                  n_jobs=N_CORES, missing=np.nan)
    return {MODEL: Model(xgb.XGBRegressor(**xgb_kw, random_state=42), target=TARGET_RANK)}


def fit_all(train_df: pd.DataFrame, verbose: bool = True) -> dict:
    models = {}
    for name, m in make_models().items():
        t0 = time.time()
        models[name] = m.fit(train_df)
        if verbose:
            print(f"  {name:16s} 학습 완료 ({time.time() - t0:.1f}s)")
    return models


def predict_all(models: dict, df: pd.DataFrame) -> dict:
    return {name: m.predict(df) for name, m in models.items()}


# ===========================================================================
# 단계 2: 단일 분할 (base vs flowmom vs retstyle vs retstyle)
# ===========================================================================

def stage_val(df: pd.DataFrame) -> None:
    header("[val] train 학습 → val 평가 (base vs flowmom vs retstyle vs retstyle)")
    train_df, val_df = select_split(df, "train"), select_split(df, "val")
    print(f"train={len(train_df)}, val={len(val_df)}")

    rows, ic_rows = [], []
    for variant in VARIANTS:
        set_variant(variant)
        print(f"\n[{variant}] feature {len(ALL_FEATURES)}개")
        preds = predict_all(fit_all(train_df), val_df)
        for name in [MODEL]:
            ic = monthly_ic(val_df["month"], preds[name], val_df[TARGET])
            ls = monthly_long_short(val_df["month"], preds[name], val_df[TARGET])
            rows.append({"variant": variant, "model": name, **summarize(ic, ls)})
            ic_rows.append(pd.DataFrame({
                "variant": variant, "model": name, "month": ic.index, "ic": ic.to_numpy(),
            }))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    val_metrics = pd.DataFrame(rows)
    val_metrics.to_csv(RESULTS_DIR / "val_metrics.csv", index=False)
    pd.concat(ic_rows).to_csv(RESULTS_DIR / "val_monthly_ic.csv", index=False)
    print("\n" + val_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    base_rank_target = val_metrics.query("variant == 'base' and model == 'xgb_rank_target'")["ic_mean"].iloc[0]
    print(f"\n재현 확인: base xgb_rank_targetVAL IC={base_rank_target:.4f} (007 로그의 0.1008과 같아야 함)")


# ===========================================================================
# 단계 3: walk-forward CV (base vs flowmom vs retstyle vs retstyle)
# ===========================================================================

def stage_cv(df: pd.DataFrame) -> None:
    header(f"[cv] walk-forward CV (val 연도 {CV_YEARS[0]}~{CV_YEARS[-1]}, base vs flowmom vs retstyle)")
    rows = []
    for year in tqdm(CV_YEARS, desc="cv fold"):
        tr = df[(df["month"] >= SPLITS["train"][0]) & (df["month"] <= f"{year - 1}-12")]
        te = df[df["month"].str.startswith(str(year))]
        for variant in VARIANTS:
            set_variant(variant)
            preds = predict_all(fit_all(tr, verbose=False), te)
            for name in [MODEL]:
                ic = monthly_ic(te["month"], preds[name], te[TARGET])
                ls = monthly_long_short(te["month"], preds[name], te[TARGET])
                for month in ic.index:
                    rows.append({"variant": variant, "model": name, "month": month,
                                 "ic": ic[month], "ls": ls.get(month, np.nan)})
        fold = pd.DataFrame([r for r in rows if r["month"].startswith(str(year))])
        mean_ic = fold.groupby(["variant", "model"])["ic"].mean().unstack(0)
        print(f"  {year}: " + ", ".join(f"Δ{v}={mean_ic.loc[MODEL, v] - mean_ic.loc[MODEL, 'base']:+.3f}" for v in COMPARE))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "cv_monthly.csv", index=False)
    print(f"\n저장 완료 -> {RESULTS_DIR}/cv_monthly.csv")


# ===========================================================================
# 단계 4: 요약
# ===========================================================================

def _heatmap(data: pd.DataFrame, path: Path, title: str) -> None:
    vals = data.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(0.75 * data.shape[1] + 3, 0.5 * data.shape[0] + 1.8))
    lim = np.nanmax(np.abs(vals)) or 1
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(data.shape[1]), data.columns, rotation=60, ha="right")
    ax.set_yticks(range(data.shape[0]), data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stage_report() -> None:
    header("[report] variant 요약 (base 대비)")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(index=COMPARE)

    val_path = RESULTS_DIR / "val_metrics.csv"
    if val_path.exists():
        val = pd.read_csv(val_path).pivot(index="model", columns="variant", values="ic_mean").loc[MODEL]
        summary["n_features"] = [len([f for c in VARIANTS[v].values() for f in c]) for v in COMPARE]
        summary["val_base"] = val["base"]
        summary["val_ic"] = [val[v] for v in COMPARE]
        summary["val_delta"] = summary["val_ic"] - summary["val_base"]

    cv_path = RESULTS_DIR / "cv_monthly.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path, dtype={"month": str})
        ic = cv[cv["model"] == MODEL].pivot_table(index="month", columns="variant", values="ic")
        for v in COMPARE:
            d = (ic[v] - ic["base"]).dropna()
            summary.loc[v, "cv_base"] = ic["base"].mean()
            summary.loc[v, "cv_ic"] = ic[v].mean()
            summary.loc[v, "cv_delta"] = d.mean()
            summary.loc[v, "cv_delta_t"] = d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 1 else np.nan
            summary.loc[v, "cv_months_improved"] = (d > 0).mean()
        cv["year"] = cv["month"].str[:4]
        yearly = cv[cv["model"] == MODEL].groupby(["variant", "year"])["ic"].mean().unstack("year")
        yearly_delta = pd.DataFrame({v: yearly.loc[v] - yearly.loc["base"] for v in COMPARE}).T
        _heatmap(yearly_delta, IMAGES_DIR / "cv_yearly_ic_delta.png",
                 "Walk-forward CV: mean Rank IC by year, variant − base")

    summary.to_csv(RESULTS_DIR / "summary.csv")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n이미지 저장 완료 -> {IMAGES_DIR}")


# ===========================================================================
# main
# ===========================================================================

def _setup_korean_font() -> str | None:
    from matplotlib import font_manager
    # matplotlib 폰트 캐시가 설치 전 목록이라 새로 설치한 한글 폰트를 못 볼 수 있어, 시스템 폰트 파일을 직접 등록
    for path in font_manager.findSystemFonts():
        if any(k in path.lower() for k in ("nanum", "cjk", "malgun")):
            font_manager.fontManager.addfont(path)
    installed = {f.name for f in font_manager.fontManager.ttflist}
    ko_font = next((c for c in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic") if c in installed), None)
    if ko_font:
        plt.rcParams["font.family"] = ko_font
        plt.rcParams["axes.unicode_minus"] = False
    return ko_font


def stage_slide_table() -> None:
    """발표용 표: variant별 안정성 지표(CV 156개월). results/cv_monthly.csv만 사용."""
    header("[slide] 안정성 지표 표")
    ko_font = _setup_korean_font()
    cv = pd.read_csv(RESULTS_DIR / "cv_monthly.csv", dtype={"month": str})
    cv = cv[cv["model"] == MODEL]
    ic = cv.pivot_table(index="month", columns="variant", values="ic")
    ls = cv.pivot_table(index="month", columns="variant", values="ls")
    yearly = ic.groupby(ic.index.str[:4]).mean()
    order = list(VARIANTS)
    ink, grid = "#0b0b0b", "#e3e2de"
    if ko_font:
        names = {"base": "base (feature 23개)", "flowmom": "flowmom (수급 모멘텀 +8)", "retstyle": "retstyle (가격식 모멘텀 +8)"}
        head = ["평균 IC", "IC-IR\n(평균/표준편차)", "적중률\n(IC>0인 월)", "최악 연도\n평균 IC", "롱숏 스프레드\n(월 평균)"]
        title = "안정성 지표 (Walk-forward CV 2009-2021, 156개월)"
    else:
        names = {"base": "base (23 features)", "flowmom": "flowmom (+8)", "retstyle": "retstyle (+8)"}
        head = ["Mean IC", "IC-IR\n(mean/std)", "Hit rate\n(IC>0)", "Worst-year\nmean IC", "Long-short\n(monthly)"]
        title = "Stability metrics (walk-forward CV 2009-2021, 156 months)"
    rows = [[
        f"{ic[v].mean():.4f}", f"{ic[v].mean() / ic[v].std():.2f}", f"{(ic[v] > 0).mean():.0%}",
        f"{yearly[v].min():.3f} ({yearly[v].idxmin()})", f"{ls[v].mean():.2%}",
    ] for v in order]

    fig, ax = plt.subplots(figsize=(11, 2.9))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, color=ink)
    tbl = ax.table(cellText=rows, rowLabels=[names[v] for v in order], colLabels=head,
                   loc="center", cellLoc="center", bbox=[0.0, 0.0, 1.0, 0.92])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(grid)
        cell.set_linewidth(0.8)
        cell.get_text().set_color(ink)
        if r == 0:
            cell.set_facecolor("#f4f3f0")
        if c == -1 and r > 0:
            cell.get_text().set_fontweight("bold")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES_DIR / "slide_stability_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료 -> {IMAGES_DIR}/slide_stability_table.png")


def stage_slide_chart() -> None:
    """발표용 도표: variant별 IC(VAL·CV)와 base 대비 Δ(CV, 월별 paired 95% CI). results/*.csv만 사용."""
    header("[slide] 발표용 도표")
    val = pd.read_csv(RESULTS_DIR / "val_metrics.csv").set_index("variant")["ic_mean"]
    cv = pd.read_csv(RESULTS_DIR / "cv_monthly.csv", dtype={"month": str})
    ic = cv[cv["model"] == MODEL].pivot_table(index="month", columns="variant", values="ic")
    order = list(VARIANTS)
    colors = {"base": "#8a8985", "flowmom": "#2a78d6", "retstyle": "#eb6834"}
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e3e2de"
    ko_font = _setup_korean_font()
    T = {
        "base": "base (feature 23개)", "flowmom": "flowmom (수급 모멘텀 +8)", "retstyle": "retstyle (가격식 모멘텀 +8)",
        "g_val": "VAL 2018-2021\n(48개월)", "g_cv": "Walk-forward CV 2009-2021\n(156개월)",
        "y1": "월별 Rank IC 평균", "t1": "feature 구성별 평균 Rank IC",
        "y2": "base 대비 ΔIC (CV, 월별 paired)", "t2": "base 대비 개선폭 (95% 신뢰구간)", "up": "개선된 월",
    } if ko_font else {
        "base": "base (23 features)", "flowmom": "flowmom (+8)", "retstyle": "retstyle (+8)",
        "g_val": "VAL 2018-2021\n(48 months)", "g_cv": "Walk-forward CV 2009-2021\n(156 months)",
        "y1": "Mean monthly Rank IC", "t1": "Mean Rank IC by feature set",
        "y2": "ΔIC vs base (CV, paired by month)", "t2": "Gain over base, 95% CI", "up": "months up",
    }
    labels = {v: T[v] for v in VARIANTS}
    print(f"폰트: {ko_font or '한글 폰트 없음 → 영어 라벨 사용 (apt install -y fonts-nanum 후 재실행)'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("white")

    # 왼쪽: 평균 Rank IC(0 기준 막대) — 차이가 실제로 얼마나 작은지 그대로 보여줌
    groups = [(T["g_val"], val), (T["g_cv"], ic.mean())]
    width = 0.24
    for gi, (_, vals) in enumerate(groups):
        for vi, v in enumerate(order):
            x = gi + (vi - 1) * (width + 0.03)
            ax1.bar(x, vals[v], width, color=colors[v], label=labels[v] if gi == 0 else None)
            ax1.text(x, vals[v] + 0.002, f"{vals[v]:.4f}", ha="center", va="bottom", fontsize=9, color=ink)
    ax1.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=10)
    ax1.set_ylim(0, 0.13)
    ax1.set_ylabel(T["y1"], color=ink2)
    ax1.set_title(T["t1"], loc="left", fontsize=12, color=ink)
    ax1.legend(frameon=False, fontsize=9, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))

    # 오른쪽: base 대비 Δ(CV) + 95% CI — 0을 가로지르면 노이즈와 구분 안 됨
    for i, v in enumerate(COMPARE):
        d = (ic[v] - ic["base"]).dropna()
        mean, se = d.mean(), d.std() / np.sqrt(len(d))
        t = mean / se
        ax2.bar(i, mean, 0.4, color=colors[v])
        ax2.errorbar(i, mean, yerr=1.96 * se, color=ink, capsize=5, lw=1.2)
        ax2.text(i, mean + 1.96 * se + 0.0006, f"Δ {mean:+.4f}\nt = {t:.2f}, {T['up']} {np.mean(d > 0):.0%}",
                 ha="center", va="bottom", fontsize=9, color=ink)
    ax2.axhline(0, color=ink2, lw=1)
    ax2.set_xticks(range(len(COMPARE)), [labels[v] for v in COMPARE], fontsize=10)
    ax2.set_xlim(-0.6, len(COMPARE) - 0.4)
    ax2.set_ylim(-0.002, 0.0125)
    ax2.set_ylabel(T["y2"], color=ink2)
    ax2.set_title(T["t2"], loc="left", fontsize=12, color=ink)

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.spines["bottom"].set_color(ink2)
        ax.tick_params(colors=ink2)
    fig.tight_layout()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES_DIR / "slide_variant_comparison.png", dpi=200)
    plt.close(fig)
    print(f"저장 완료 -> {IMAGES_DIR}/slide_variant_comparison.png")


def main() -> None:
    if "slide" in sys.argv[1:]:  # 결과 CSV가 이미 있을 때 도표만 다시 그림
        stage_slide_chart()
        stage_slide_table()
        return
    stage_features()
    df = load_panel()
    stage_val(df)
    stage_cv(df)
    stage_report()
    stage_slide_chart()
    stage_slide_table()


if __name__ == "__main__":
    main()
