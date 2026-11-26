"""
feature_progression.py: 018 실험 — 005 → 006 → 017로 feature를 늘려 온 과정을 같은 조건(xgb_rank_target,
walk-forward CV, 하이퍼파라미터 고정)에서 다시 비교한다. feature 구성만 다르고 모델·분할·설정은 모두 같다.

실행 순서:
  1. 종목×월 feature 패널 생성 (005 수익률 lag + 006/007 feature + 017 수급 모멘텀). features/panel.csv.gz가 있으면 재사용
  2. feature 구성별 walk-forward CV (val 연도 2009~2021, 매 fold 2001-01부터 직전 연도까지 학습)
  3. 요약표(results/summary.csv)와 플롯(images/). 같은 fold에서 feature·그룹별 순열 중요도도 계산

사용법:
    uv run python members/wooooooooook/018_feature_progression/feature_progression.py
    uv run python members/wooooooooook/018_feature_progression/feature_progression.py plot   # 결과 CSV로 플롯만
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import font_manager
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

TRAIN_START = "2001-01"
CV_YEARS = list(range(2009, 2022))
RETURN_PERIODS = ["warmup_2000", "train_2001_2021", "test_2022_2025"]
OUTPUT_START, OUTPUT_END = "2001-01", "2025-12"
EXCLUDED_CODES = {"004730", "217590"}

# 고정 설정 (007 xgb_rank_target과 동일)
N_TREES, DEPTH, LR, SEED = 500, 4, 0.05, 42
N_CORES = os.cpu_count() or 1

MACRO_COLS = ["base_rate", "fx_change", "term_spread", "cpi_yoy", "kospi_rv"]
RETURN_005 = ["return_lag1", "return_lag3", "return_lag6", "return_lag12"]
FLOW_4 = [
    "individual_net_intensity", "individual_trade_intensity",
    "foreign_net_intensity", "foreign_trade_intensity",
]
FEATURES_006 = [
    "ret_1m", "ret_3m", "mom_12_2", "vol_12m",
    "log_mktcap", "log_price", "share_issuance_12m",
    "turnover_1m", "abnormal_turnover", "amihud_illiq",
    *FLOW_4,
    "foreign_own_ratio", "foreign_own_chg_1m", "foreign_own_chg_3m", "foreign_limit_exhaustion",
    *MACRO_COLS,
]
FLOW_MOM_COLS = [
    f"{tag}_{name}"
    for tag in ("individual", "foreign")
    for name in ("net_3m", "net_12m", "net_accel", "trade_abnormal")
]
FEATURE_SETS = {
    "005_step4": RETURN_005 + FLOW_4 + MACRO_COLS,
    "006": FEATURES_006,
    "017_flowmom": FEATURES_006 + FLOW_MOM_COLS,
}
PANEL_FEATURES = list(dict.fromkeys(f for cols in FEATURE_SETS.values() for f in cols))
TARGET = "target_next"
TARGET_RANK = "target_rank"

# 순열 중요도: 테스트 연도에서 feature(또는 그룹)를 섞었을 때 월별 Rank IC 평균이 얼마나 떨어지는지
N_PERM_REPEATS = 3
PERM_SEED = 0
TOP_N_FEATURES = 12
HIGHLIGHT_YEAR = "2017"   # 월별 차이 도표에서 12개월을 확대해 보여 주는 연도

GROUPS = {
    "수익률": RETURN_005 + ["ret_1m", "ret_3m", "mom_12_2", "vol_12m"],
    "규모": ["log_mktcap", "log_price", "share_issuance_12m"],
    "유동성": ["turnover_1m", "abnormal_turnover", "amihud_illiq"],
    "개인 수급": ["individual_net_intensity", "individual_trade_intensity"],
    "외국인 수급": ["foreign_net_intensity", "foreign_trade_intensity"],
    "외국인 보유": ["foreign_own_ratio", "foreign_own_chg_1m", "foreign_own_chg_3m", "foreign_limit_exhaustion"],
    "수급 누적·가속(017)": FLOW_MOM_COLS,
    "매크로": MACRO_COLS,
}
GROUP_EN = {
    "수익률": "Return", "규모": "Size", "유동성": "Liquidity", "개인 수급": "Individual flow",
    "외국인 수급": "Foreign flow", "외국인 보유": "Foreign holding",
    "수급 누적·가속(017)": "Flow accum/accel (017)", "매크로": "Macro",
}
GROUP_OF = {f: g for g, cols in GROUPS.items() for f in cols}
GROUP_COLORS = dict(zip(GROUPS, [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
]))
FEATURE_KO = {
    "return_lag1": "수익률 lag 1개월", "return_lag3": "수익률 lag 3개월 누적",
    "return_lag6": "수익률 lag 6개월 누적", "return_lag12": "수익률 lag 12개월 누적",
    "ret_1m": "당월 수익률", "ret_3m": "3개월 누적 수익률", "mom_12_2": "12-2 모멘텀", "vol_12m": "12개월 변동성",
    "log_mktcap": "시가총액(로그)", "log_price": "주가(로그)", "share_issuance_12m": "12개월 주식수 증가율",
    "turnover_1m": "회전율", "abnormal_turnover": "이상 회전율", "amihud_illiq": "비유동성(Amihud)",
    "individual_net_intensity": "개인 순매수강도", "individual_trade_intensity": "개인 거래강도",
    "foreign_net_intensity": "외국인 순매수강도", "foreign_trade_intensity": "외국인 거래강도",
    "foreign_own_ratio": "외국인 보유비율", "foreign_own_chg_1m": "외국인 보유 1개월 변화",
    "foreign_own_chg_3m": "외국인 보유 3개월 변화", "foreign_limit_exhaustion": "외국인 한도소진률",
    "base_rate": "기준금리", "fx_change": "환율 변화", "term_spread": "장단기 금리차",
    "cpi_yoy": "물가상승률", "kospi_rv": "코스피 실현변동성",
    "individual_net_3m": "개인 순매수 3개월 합", "individual_net_12m": "개인 순매수 12개월 합",
    "individual_net_accel": "개인 순매수 가속", "individual_trade_abnormal": "개인 거래강도 이상치",
    "foreign_net_3m": "외국인 순매수 3개월 합", "foreign_net_12m": "외국인 순매수 12개월 합",
    "foreign_net_accel": "외국인 순매수 가속", "foreign_trade_abnormal": "외국인 거래강도 이상치",
}

# 017 로그의 CV IC(재현 확인용)
REF_017_CV = {"006": 0.1055, "017_flowmom": 0.1089}

COLORS = {"005_step4": "#2a78d6", "006": "#eb6834", "017_flowmom": "#1baf7a"}

# 비교 쌍: (앞 구성, 뒤 구성). 플롯은 쌍마다 images/<쌍 이름>/ 에 따로 저장
COMPARISONS = {
    "005_to_006": ["005_step4", "006"],
    "006_to_017": ["006", "017_flowmom"],
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ===========================================================================
# 평가 함수 (007과 동일)
# ===========================================================================

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


# ===========================================================================
# 단계 1: feature 패널
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

    # 005: t-1월부터 거슬러 올라간 수익률 lag (당월 수익률 제외)
    f["return_lag1"] = ret.shift(1)
    for k in (3, 6, 12):
        f[f"return_lag{k}"] = np.exp(log_gross.shift(1).rolling(k, min_periods=k).sum()) - 1

    # 006/007
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

    # 017 flowmom: 순매수 누적·가속, 거래강도 이상치. 외국인 결측은 채우지 않고 NaN 전파
    for tag in ("individual", "foreign"):
        net, trade = f[f"{tag}_net_intensity"], f[f"{tag}_trade_intensity"]
        f[f"{tag}_net_3m"] = net.rolling(3, min_periods=3).sum()
        f[f"{tag}_net_12m"] = net.rolling(12, min_periods=12).sum()
        f[f"{tag}_net_accel"] = net - net.shift(1).rolling(3, min_periods=3).mean()
        f[f"{tag}_trade_abnormal"] = trade / _pos(trade.shift(1).rolling(12, min_periods=6).mean())

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
    print("feature 구성별 개수:", {k: len(v) for k, v in FEATURE_SETS.items()})


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, dtype={"month": str, "code": str})
    df[TARGET_RANK] = df.groupby("month")[TARGET].rank(pct=True) - 0.5
    return df


# ===========================================================================
# 단계 2: walk-forward CV (feature 구성별)
# ===========================================================================

def fit_model(train_df: pd.DataFrame, features: list[str]) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=N_TREES, max_depth=DEPTH, learning_rate=LR,
        n_jobs=N_CORES, missing=np.nan, random_state=SEED,
    )
    model.fit(train_df[features], train_df[TARGET_RANK].to_numpy())
    return model


def permuted_ic(model, X: pd.DataFrame, month_groups: list, months, actual, cols: list[str],
                macro: bool, rng: np.random.Generator) -> float:
    """cols를 섞은 뒤의 월별 Rank IC 평균(반복 평균). 종목 feature는 같은 달 안에서 행을 섞고(그룹이면 같은
    순열을 공유해 그룹 내 관계는 유지), 매크로는 달마다 값이 같아서 월 단위로 다른 달의 값과 바꾼다."""
    orig = X[cols].to_numpy().copy()
    out = []
    for _ in range(N_PERM_REPEATS):
        new = orig.copy()
        if macro:
            perm = rng.permutation(len(month_groups))
            first_rows = np.array([g[0] for g in month_groups])
            for gi, g in enumerate(month_groups):
                new[g, :] = orig[first_rows[perm[gi]], :]
        else:
            for g in month_groups:
                new[g, :] = orig[rng.permutation(g), :]
        X[cols] = new
        out.append(monthly_ic(months, model.predict(X), actual).mean())
    X[cols] = orig
    return float(np.mean(out))


def stage_cv(df: pd.DataFrame) -> None:
    header(f"[cv] walk-forward CV (val 연도 {CV_YEARS[0]}~{CV_YEARS[-1]}, feature 구성 {len(FEATURE_SETS)}개)")
    rows, feat_rows, group_rows = [], [], []
    rng = np.random.default_rng(PERM_SEED)
    for year in tqdm(CV_YEARS, desc="cv fold"):
        tr = df[(df["month"] >= TRAIN_START) & (df["month"] <= f"{year - 1}-12")]
        te = df[df["month"].str.startswith(str(year))].reset_index(drop=True)
        months, actual = te["month"].to_numpy(), te[TARGET].to_numpy()
        month_groups = [np.flatnonzero(months == m) for m in sorted(set(months))]
        for name, features in FEATURE_SETS.items():
            model = fit_model(tr, features)
            X = te[features].copy()
            pred = np.asarray(model.predict(X), dtype=float)
            ic = monthly_ic(months, pred, actual)
            ls = monthly_long_short(months, pred, actual)
            for month in ic.index:
                rows.append({"feature_set": name, "month": month, "ic": ic[month], "ls": ls.get(month, np.nan)})

            base_ic = ic.mean()
            for f in features:
                drop = base_ic - permuted_ic(model, X, month_groups, months, actual, [f], f in MACRO_COLS, rng)
                feat_rows.append({"feature_set": name, "year": year, "feature": f, "group": GROUP_OF[f], "importance": drop})
            for g, cols in GROUPS.items():
                cols = [c for c in cols if c in features]
                if cols:
                    drop = base_ic - permuted_ic(model, X, month_groups, months, actual, cols, g == "매크로", rng)
                    group_rows.append({"feature_set": name, "year": year, "group": g, "importance": drop})
        fold = pd.DataFrame([r for r in rows if r["month"].startswith(str(year))])
        mean_ic = fold.groupby("feature_set")["ic"].mean().reindex(list(FEATURE_SETS))
        print(f"  {year}: " + ", ".join(f"{k}={v:.3f}" for k, v in mean_ic.items()))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "cv_monthly.csv", index=False)
    pd.DataFrame(feat_rows).to_csv(RESULTS_DIR / "perm_importance_feature.csv", index=False)
    pd.DataFrame(group_rows).to_csv(RESULTS_DIR / "perm_importance_group.csv", index=False)
    print(f"\n저장 완료 -> {RESULTS_DIR}/cv_monthly.csv, perm_importance_feature.csv, perm_importance_group.csv")


# ===========================================================================
# 단계 3: 요약 + 플롯
# ===========================================================================

def _setup_korean_font() -> str | None:
    # matplotlib 폰트 캐시가 설치 전 목록일 수 있어 시스템 폰트 파일을 직접 등록
    for path in font_manager.findSystemFonts():
        if any(k in path.lower() for k in ("nanum", "cjk", "malgun")):
            font_manager.fontManager.addfont(path)
    installed = {f.name for f in font_manager.fontManager.ttflist}
    name = next((c for c in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic") if c in installed), None)
    if name:
        plt.rcParams["font.family"] = name
        plt.rcParams["axes.unicode_minus"] = False
    return name


def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2)


def make_labels(ko) -> dict:
    return {
        "005_step4": "수익률 모멘텀 + 투자자별 수급 강도\n+ 매크로 (13)" if ko
        else "return momentum + investor flow\n+ macro (13)",
        "006": "수익률 feature 재구성\n+ 규모·유동성·외국인 보유 (23)" if ko
        else "return features rebuilt\n+ size, liquidity, foreign holding (23)",
        "017_flowmom": "+ 수급 누적·가속·이상치 (31)" if ko
        else "+ flow accumulation/accel/abnormal (31)",
    }


COMPARISON_TITLES = {
    "005_to_006": ("005 → 006: feature 확장", "005 → 006: feature expansion"),
    "006_to_017": ("006 → 017 flowmom: 수급 모멘텀 추가", "006 → 017 flowmom: flow momentum added"),
}


def _comparison_title(key: str, ko) -> str:
    return COMPARISON_TITLES[key][0 if ko else 1]


def _paired(ic: pd.DataFrame, a: str, b: str):
    d = (ic[b] - ic[a]).dropna()
    return d.mean(), d.mean() / (d.std() / np.sqrt(len(d))), (d > 0).mean()


def plot_ic_bars(ic: pd.DataFrame, names: list, out_dir: Path, ko, labels: dict, title: str) -> None:
    """두 구성의 CV 평균 Rank IC(95% CI)와 월별 paired 차이."""
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    fig.patch.set_facecolor("white")
    for i, n in enumerate(names):
        mean, se = ic[n].mean(), ic[n].std() / np.sqrt(ic[n].count())
        ax.bar(i, mean, 0.5, color=COLORS[n])
        ax.errorbar(i, mean, yerr=1.96 * se, color=INK, capsize=5, lw=1.2)
        ax.text(i, mean + 1.96 * se + 0.002, f"{mean:.4f}", ha="center", va="bottom", fontsize=11, color=INK)
    delta, t, share_up = _paired(ic, names[0], names[1])
    ax.text(0.5, 0.97, (f"Δ {delta:+.4f}   t = {t:.2f}   개선된 월 {share_up:.0%}" if ko
                        else f"Δ {delta:+.4f}   t = {t:.2f}   months up {share_up:.0%}"),
            transform=ax.transAxes, ha="center", va="top", fontsize=11, color=INK)
    ax.set_xticks(range(len(names)), [labels[n] for n in names], fontsize=10)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, max(ic[names].mean()) * 1.35)
    ax.set_ylabel("Walk-forward CV 월별 Rank IC 평균 (95% CI)" if ko else "Walk-forward CV mean monthly Rank IC (95% CI)",
                  color=INK2)
    ax.set_title(title, loc="left", fontsize=13, color=INK)
    _style(ax)
    fig.text(0.01, 0.01, "모델·하이퍼파라미터·분할 동일, feature 구성만 다름" if ko
             else "Same model, hyperparameters and splits; only the feature set differs", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir / "cv_ic.png", dpi=200)
    plt.close(fig)


def plot_yearly(yearly: pd.DataFrame, names: list, out_dir: Path, ko, labels: dict, title: str) -> None:
    """연도별 평균 IC(두 구성)와 연도별 차이(b − a)."""
    years = [int(y) for y in yearly.index]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True, gridspec_kw={"height_ratios": [2, 1.1]})
    fig.patch.set_facecolor("white")
    for n in names:
        ax1.plot(years, yearly[n], color=COLORS[n], lw=2, marker="o", ms=5, label=labels[n].replace("\n", " "))
    ax1.axhline(0, color=INK2, lw=1)
    ax1.set_ylabel("연도별 월별 Rank IC 평균" if ko else "Yearly mean of monthly Rank IC", color=INK2)
    ax1.set_title(title, loc="left", fontsize=13, color=INK)
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax1)

    a, b = names
    diff = yearly[b] - yearly[a]
    ax2.bar(years, diff, 0.6, color=COLORS[b])
    for x, v in zip(years, diff):
        ax2.text(x, v + (0.0006 if v >= 0 else -0.0006), f"{v:+.3f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
    ax2.axhline(0, color=INK2, lw=1)
    ax2.set_xticks(years)
    ax2.set_ylabel("연도별 차이 (뒤 - 앞)" if ko else "Yearly difference (later - earlier)", color=INK2)
    ax2.margins(y=0.25)
    _style(ax2)
    fig.tight_layout()
    fig.savefig(out_dir / "cv_yearly_ic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_delta(ic: pd.DataFrame, names: list, out_dir: Path, ko, labels: dict, title: str) -> None:
    """월별 차이(b - a) 전체 156개월과, HIGHLIGHT_YEAR 12개월 확대(두 구성의 월별 IC와 차이)."""
    a, b = names
    d = (ic[b] - ic[a]).dropna()
    x = pd.to_datetime(d.index, format="%Y-%m")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1.3, 1.2, 1]})
    fig.patch.set_facecolor("white")

    ax1.bar(x, d.to_numpy(), width=25, color=COLORS[b], label="월별 차이" if ko else "monthly difference")
    ax1.plot(x, d.rolling(12, min_periods=12).mean().to_numpy(), color=INK, lw=2,
             label="12개월 이동평균" if ko else "12-month mean")
    ax1.axvspan(pd.Timestamp(f"{HIGHLIGHT_YEAR}-01-01"), pd.Timestamp(f"{HIGHLIGHT_YEAR}-12-31"),
                color="#eda100", alpha=0.15, label=f"{HIGHLIGHT_YEAR}")
    ax1.axhline(0, color=INK2, lw=1)
    ax1.set_ylabel("월별 IC 차이 (뒤 - 앞)" if ko else "Monthly IC difference (later - earlier)", color=INK2)
    ax1.set_title(title + (f" — 월별 차이 ({len(d)}개월, 플러스 {int((d > 0).sum())}개월)" if ko
                           else f" — monthly difference ({len(d)} months, {int((d > 0).sum())} positive)"),
                  loc="left", fontsize=13, color=INK)
    ax1.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    _style(ax1)

    months = [m for m in d.index if m.startswith(HIGHLIGHT_YEAR)]
    pos = np.arange(len(months))
    ticks = [f"{int(m[5:7])}월" if ko else m[5:7] for m in months]
    for i, n in enumerate(names):
        ax2.bar(pos + (i - 0.5) * 0.38, ic.loc[months, n], 0.36, color=COLORS[n], label=labels[n].replace("\n", " "))
    ax2.axhline(0, color=INK2, lw=1)
    ax2.set_xticks(pos, ticks)
    ax2.set_ylabel("월별 Rank IC" if ko else "Monthly Rank IC", color=INK2)
    ax2.set_title(f"{HIGHLIGHT_YEAR}년 월별 IC" if ko else f"Monthly IC in {HIGHLIGHT_YEAR}",
                  loc="left", fontsize=12, color=INK)
    ax2.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax2)

    diff = d.loc[months]
    ax3.bar(pos, diff.to_numpy(), 0.55, color=COLORS[b])
    for p, v in zip(pos, diff):
        ax3.text(p, v + (0.001 if v >= 0 else -0.001), f"{v:+.3f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
    ax3.axhline(0, color=INK2, lw=1)
    ax3.set_xticks(pos, ticks)
    ax3.margins(y=0.25)
    ax3.set_ylabel("차이 (뒤 - 앞)" if ko else "Difference (later - earlier)", color=INK2)
    ax3.set_title(f"{HIGHLIGHT_YEAR}년 월별 차이 (평균 {diff.mean():+.4f}, 플러스 {int((diff > 0).sum())}/12개월)" if ko
                  else f"Difference in {HIGHLIGHT_YEAR} (mean {diff.mean():+.4f}, {int((diff > 0).sum())}/12 positive)",
                  loc="left", fontsize=12, color=INK)
    _style(ax3)
    fig.tight_layout()
    fig.savefig(out_dir / "cv_monthly_delta.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_importance_feature(feat: pd.DataFrame, names: list, out_dir: Path, ko, labels: dict, title: str) -> None:
    """feature별 순열 중요도(구성별 패널, 그룹별 색, 상위 N개)."""
    feat = feat[feat["feature_set"].isin(names)]
    gname = (lambda g: g) if ko else (lambda g: GROUP_EN[g])
    xlabel = ("feature를 섞었을 때 Rank IC 하락량 (클수록 기여가 큼)" if ko
              else "Rank IC drop when permuted (larger = more important)")
    fig, axes = plt.subplots(1, len(names), figsize=(5.8 * len(names), 6.4), sharex=True)
    fig.patch.set_facecolor("white")
    for ax, n in zip(np.atleast_1d(axes), names):
        d = feat[feat["feature_set"] == n].groupby("feature")["importance"].agg(["mean", "std", "count"])
        d["se"] = d["std"] / np.sqrt(d["count"])
        d = d.sort_values("mean", ascending=False).head(TOP_N_FEATURES)[::-1]
        ys = np.arange(len(d))
        ax.barh(ys, d["mean"], 0.62, color=[GROUP_COLORS[GROUP_OF[f]] for f in d.index])
        ax.errorbar(d["mean"], ys, xerr=d["se"], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.set_yticks(ys, [FEATURE_KO.get(f, f) if ko else f for f in d.index], fontsize=9.5)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(labels[n], loc="left", fontsize=11, color=INK)
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
        ax.set_facecolor("white")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(INK2)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2)
    present = [g for g in GROUPS if g in set(feat["group"])]
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in present]
    fig.legend(handles, [gname(g) for g in present], frameon=False, fontsize=10, ncol=len(present),
               loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(title + (" — feature별 순열 중요도 (CV 평균 ± 표준오차)" if ko
                          else " — permutation importance by feature (CV mean ± SE)"),
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_dir / "permutation_importance_feature.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_importance_group(grp: pd.DataFrame, names: list, out_dir: Path, ko, labels: dict, title: str) -> None:
    """그룹별 순열 중요도(그룹 전체를 한꺼번에 섞음)."""
    grp = grp[grp["feature_set"].isin(names)]
    gname = (lambda g: g) if ko else (lambda g: GROUP_EN[g])
    xlabel = ("그룹을 섞었을 때 Rank IC 하락량 (클수록 기여가 큼)" if ko
              else "Rank IC drop when group permuted (larger = more important)")
    present = [g for g in GROUPS if g in set(grp["group"])]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    fig.patch.set_facecolor("white")
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        d = grp[grp["feature_set"] == n].groupby("group")["importance"].agg(["mean", "std", "count"]).reindex(present)
        se = d["std"] / np.sqrt(d["count"])
        x = np.arange(len(present)) + (i - (len(names) - 1) / 2) * width
        ax.bar(x, d["mean"].fillna(0), width * 0.9, color=COLORS[n], label=labels[n].replace("\n", " "))
        ax.errorbar(x, d["mean"].fillna(0), yerr=se.fillna(0), fmt="none", ecolor=INK, capsize=3, lw=1)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(np.arange(len(present)), [gname(g) for g in present], fontsize=10)
    ax.set_ylabel(xlabel, color=INK2, fontsize=9)
    ax.set_title(title + (" — 그룹별 순열 중요도 (CV 평균 ± 표준오차)" if ko
                          else " — permutation importance by group (CV mean ± SE)"),
                 loc="left", fontsize=13, color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=1, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "permutation_importance_group.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def stage_importance() -> None:
    header("[importance] 순열 중요도 플롯 (비교 쌍별)")
    feat_path, group_path = RESULTS_DIR / "perm_importance_feature.csv", RESULTS_DIR / "perm_importance_group.csv"
    if not (feat_path.exists() and group_path.exists()):
        print("순열 중요도 결과 CSV가 없음 → 전체 실행(stage_cv) 필요")
        return
    ko = _setup_korean_font()
    labels = make_labels(ko)
    feat, grp = pd.read_csv(feat_path), pd.read_csv(group_path)
    for key, names in COMPARISONS.items():
        out_dir = IMAGES_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        title = _comparison_title(key, ko)
        plot_importance_feature(feat, names, out_dir, ko, labels, title)
        plot_importance_group(grp, names, out_dir, ko, labels, title)
        print(f"저장 완료 -> {out_dir}/permutation_importance_feature.png, permutation_importance_group.png")


def stage_report() -> None:
    header("[report] feature 구성별 요약")
    cv = pd.read_csv(RESULTS_DIR / "cv_monthly.csv", dtype={"month": str})
    ic = cv.pivot_table(index="month", columns="feature_set", values="ic")[list(FEATURE_SETS)]
    ls = cv.pivot_table(index="month", columns="feature_set", values="ls")[list(FEATURE_SETS)]
    yearly = ic.groupby(ic.index.str[:4]).mean()

    summary = pd.DataFrame(index=list(FEATURE_SETS))
    summary["n_features"] = [len(v) for v in FEATURE_SETS.values()]
    summary["cv_ic"] = ic.mean()
    summary["ic_ir"] = ic.mean() / ic.std()
    summary["hit_rate"] = (ic > 0).mean()
    summary["worst_year_ic"] = yearly.min()
    summary["worst_year"] = yearly.idxmin()
    summary["ls_spread"] = ls.mean()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "summary.csv")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # 비교 쌍별 요약: 앞 구성 → 뒤 구성의 평균 IC 차이, 월별 paired t, 개선된 월·연도 비율
    comp_rows = []
    for key, (a, b) in COMPARISONS.items():
        delta, t, share_up = _paired(ic, a, b)
        yd = yearly[b] - yearly[a]
        comp_rows.append({"comparison": key, "from": a, "to": b, "ic_from": ic[a].mean(), "ic_to": ic[b].mean(),
                          "delta": delta, "delta_t": t, "months_up": share_up, "years_up": (yd > 0).mean(),
                          "n_months": ic[[a, b]].dropna().shape[0]})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(RESULTS_DIR / "comparison_summary.csv", index=False)
    print("\n비교 쌍별 요약:")
    print(comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n017 로그와 대조(CV IC):")
    for name, ref in REF_017_CV.items():
        print(f"  {name}: 이번 {summary.loc[name, 'cv_ic']:.4f} / 017 로그 {ref:.4f}")

    ko = _setup_korean_font()
    print(f"폰트: {ko or '한글 폰트 없음 → 영어 라벨 사용'}")
    labels = make_labels(ko)
    for key, names in COMPARISONS.items():
        out_dir = IMAGES_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        title = _comparison_title(key, ko)
        plot_ic_bars(ic, names, out_dir, ko, labels, title)
        plot_yearly(yearly, names, out_dir, ko, labels, title)
        plot_monthly_delta(ic, names, out_dir, ko, labels, title)
    print(f"\n이미지 저장 완료 -> {IMAGES_DIR}/{{{','.join(COMPARISONS)}}}")


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    if "plot" in sys.argv[1:]:
        stage_report()
        stage_importance()
        return
    stage_features()
    df = load_panel()
    stage_cv(df)
    stage_report()
    stage_importance()


if __name__ == "__main__":
    main()
