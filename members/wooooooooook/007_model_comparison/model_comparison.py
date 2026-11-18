"""
model_comparison.py: 007 실험 — 006과 같은 재무·거래 feature 23개로 여러 모델을 학습해 비교하고,
모델별로 각 팩터가 Rank IC에 얼마나 기여하는지(permutation importance) 측정한다.

실행 순서:
  1. 종목×월 feature 패널 생성 (006과 동일한 정의). features/panel.csv.gz가 있으면 재사용
  2. train(2001-2017) 학습 → val(2018-2021) 평가 + 모델별 permutation importance
  3. walk-forward CV (val 연도 2009~2021, 매 fold 2001-01부터 직전 연도까지 학습)
  4. 비교 테이블·히트맵 이미지 생성

사용법:
    cd members/wooooooooook/007_model_comparison
    python3 model_comparison.py
"""

import os
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
STOCK_FEATURES = [f for f in ALL_FEATURES if f not in MACRO_COLS]
TARGET = "target_next"
TARGET_RANK = "target_rank"  # 월별 횡단면 순위 (-0.5 ~ 0.5)

N_CORES = os.cpu_count() or 1
SEED = 42
N_PERM_REPEATS = 3

# 트리 부스팅 공통 크기 (006 XGBoost와 동일)
N_TREES, DEPTH, LR = 500, 4, 0.05


# ===========================================================================
# 공용 함수
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
    """월별 횡단면 Spearman 상관 (= Rank IC)."""
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
    """월별 예측 상위 10% 평균 수익률 - 하위 10% 평균 수익률 (동일가중)."""
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
# 단계 1: feature 패널 (006과 동일한 정의)
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

    f[TARGET] = ret.shift(-1)
    return f, mcap, months, codes


def stage_features() -> None:
    header("[features] feature 패널 생성")
    if FEATURES_PATH.exists():
        print(f"이미 존재 -> {FEATURES_PATH} (feature 정의를 바꿨다면 이 파일을 지우고 실행)")
        return

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
    panel = panel[["month", "code"] + ALL_FEATURES + [TARGET]]

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(FEATURES_PATH, index=False)
    print(f"저장 완료 -> {FEATURES_PATH}, 행수={len(panel)}")


# ===========================================================================
# 모델
#   - 트리 계열: 원본 feature 그대로 (결측은 각 라이브러리 처리 / sklearn 트리는 train 중앙값 대체)
#   - ridge / mlp: 종목 feature는 월별 횡단면 순위(-0.5~0.5, 결측=0), 매크로는 train 평균·표준편차로
#     표준화. 타겟도 월별 순위 — 이상치에 민감한 모델이라 원값 수익률 대신 사용
#   - xgb_rank_target: XGBoost에 타겟만 월별 순위로 바꿈
#   - xgb_pairwise: XGBRanker(rank:pairwise), 월=query, 라벨=월별 수익률 10분위
#   - ensemble: 나머지 모델 예측을 월별 순위로 바꿔 평균
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
    def __init__(self, est, target: str = TARGET, rank_input: bool = False, impute: bool = False):
        self.est, self.target, self.rank_input, self.impute = est, target, rank_input, impute

    def _X(self, df: pd.DataFrame, fit: bool = False):
        if self.rank_input:
            if fit:
                self.scaler = RankScaler().fit(df)
            return self.scaler.transform(df)
        X = df[ALL_FEATURES]
        if self.impute:
            from sklearn.impute import SimpleImputer
            if fit:
                self.imputer = SimpleImputer(strategy="median").fit(X)
            return self.imputer.transform(X)
        return X

    def fit(self, df: pd.DataFrame) -> "Model":
        self.est.fit(self._X(df, fit=True), df[self.target].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.est.predict(self._X(df)), dtype=float)


class XGBPairwise:
    def fit(self, df: pd.DataFrame) -> "XGBPairwise":
        d = df.sort_values("month", kind="stable")
        pct = d.groupby("month")[TARGET].rank(pct=True).to_numpy()
        label = np.minimum((pct * 10).astype(int), 9)
        qid = pd.factorize(d["month"])[0]
        self.est = xgb.XGBRanker(
            objective="rank:pairwise",
            lambdarank_pair_method="mean",
            lambdarank_num_pair_per_sample=8,
            n_estimators=N_TREES, max_depth=DEPTH, learning_rate=LR,
            random_state=SEED, n_jobs=N_CORES, missing=np.nan,
        )
        self.est.fit(d[ALL_FEATURES], label, qid=qid)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.est.predict(df[ALL_FEATURES]), dtype=float)


def make_models() -> dict:
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor

    xgb_kw = dict(n_estimators=N_TREES, max_depth=DEPTH, learning_rate=LR,
                  random_state=SEED, n_jobs=N_CORES, missing=np.nan)
    forest_kw = dict(n_estimators=300, min_samples_leaf=100, max_features=0.5,
                     random_state=SEED, n_jobs=N_CORES)
    return {
        "ridge": Model(Ridge(alpha=1.0), target=TARGET_RANK, rank_input=True),
        "mlp": Model(
            MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=1024,
                         learning_rate_init=1e-3, max_iter=100, early_stopping=True,
                         n_iter_no_change=10, random_state=SEED),
            target=TARGET_RANK, rank_input=True,
        ),
        "random_forest": Model(RandomForestRegressor(**forest_kw), impute=True),
        "extra_trees": Model(ExtraTreesRegressor(**forest_kw), impute=True),
        "xgb_reg": Model(xgb.XGBRegressor(**xgb_kw)),
        "xgb_rank_target": Model(xgb.XGBRegressor(**xgb_kw), target=TARGET_RANK),
        "xgb_pairwise": XGBPairwise(),
        "lgbm_reg": Model(lgb.LGBMRegressor(
            n_estimators=N_TREES, max_depth=DEPTH, num_leaves=2 ** DEPTH - 1, learning_rate=LR,
            min_child_samples=100, random_state=SEED, n_jobs=N_CORES, verbose=-1,
        )),
        "catboost_reg": Model(CatBoostRegressor(
            iterations=N_TREES, depth=DEPTH, learning_rate=LR,
            random_seed=SEED, thread_count=N_CORES, verbose=0,
        )),
    }


MODEL_ORDER = [
    "ridge", "mlp", "random_forest", "extra_trees",
    "xgb_reg", "xgb_rank_target", "xgb_pairwise", "lgbm_reg", "catboost_reg", "ensemble",
]


def fit_all(train_df: pd.DataFrame, verbose: bool = True) -> dict:
    models = {}
    for name, m in make_models().items():
        t0 = time.time()
        models[name] = m.fit(train_df)
        if verbose:
            print(f"  {name:16s} 학습 완료 ({time.time() - t0:.1f}s)")
    return models


def ensemble_pred(preds: dict, months) -> np.ndarray:
    months = np.asarray(months)
    ranks = [pd.Series(p).groupby(months).rank(pct=True).to_numpy() for p in preds.values()]
    return np.mean(ranks, axis=0)


def predict_all(models: dict, df: pd.DataFrame) -> dict:
    preds = {name: m.predict(df) for name, m in models.items()}
    preds["ensemble"] = ensemble_pred(preds, df["month"])
    return preds


# ===========================================================================
# 단계 2: 단일 분할 평가 + permutation importance
#   importance = 기준 val IC - 해당 feature(그룹)를 섞은 뒤 val IC  (클수록 그 팩터에 의존)
#   - 종목 feature: 전체 행을 무작위로 섞음
#   - 매크로: 그 달 전 종목에 같은 값이라는 구조를 유지하도록 "월 단위"로 섞음
#   - 그룹: 그룹 내 feature를 같은 순서로 함께 섞음 → 그룹 내 상관 feature끼리 서로 대체하는 효과 제거
# ===========================================================================

def _permute(df: pd.DataFrame, cols: list[str], rng) -> pd.DataFrame:
    out = df.copy()
    if all(c in MACRO_COLS for c in cols):
        months = df["month"].unique()
        mapping = dict(zip(months, rng.permutation(months)))
        src = df.groupby("month")[cols].first()
        out[cols] = src.loc[df["month"].map(mapping)].to_numpy()
    else:
        out[cols] = df[cols].to_numpy()[rng.permutation(len(df))]
    return out


def stage_val(df: pd.DataFrame) -> None:
    header("[val] train 학습 → val 평가")
    train_df, val_df = select_split(df, "train"), select_split(df, "val")
    print(f"train={len(train_df)}, val={len(val_df)}")
    models = fit_all(train_df)
    preds = predict_all(models, val_df)

    rows, ic_rows = [], []
    base_ic = {}
    for name in MODEL_ORDER:
        ic = monthly_ic(val_df["month"], preds[name], val_df[TARGET])
        ls = monthly_long_short(val_df["month"], preds[name], val_df[TARGET])
        s = summarize(ic, ls)
        base_ic[name] = s["ic_mean"]
        rows.append({"model": name, **s})
        ic_rows.append(pd.DataFrame({"model": name, "month": ic.index, "ic": ic.to_numpy()}))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    val_metrics = pd.DataFrame(rows)
    val_metrics.to_csv(RESULTS_DIR / "val_metrics.csv", index=False)
    pd.concat(ic_rows).to_csv(RESULTS_DIR / "val_monthly_ic.csv", index=False)
    print("\n" + val_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    header("[val] permutation importance (ΔIC)")
    rng = np.random.default_rng(SEED)
    targets = [("feature", f, [f]) for f in ALL_FEATURES]
    targets += [("group", g, cols) for g, cols in FEATURE_GROUPS.items()]
    records = []
    for kind, name, cols in tqdm(targets, desc="permutation"):
        drops = {m: [] for m in MODEL_ORDER}
        for _ in range(N_PERM_REPEATS):
            perm_df = _permute(val_df, cols, rng)
            for m, p in predict_all(models, perm_df).items():
                drops[m].append(base_ic[m] - monthly_ic(perm_df["month"], p, perm_df[TARGET]).mean())
        for m, v in drops.items():
            records.append({"kind": kind, "name": name, "model": m, "delta_ic": float(np.mean(v))})
    imp = pd.DataFrame(records)
    imp.to_csv(RESULTS_DIR / "importance.csv", index=False)

    group_tbl = imp[imp["kind"] == "group"].pivot(index="model", columns="name", values="delta_ic")
    print(group_tbl.loc[MODEL_ORDER].to_string(float_format=lambda x: f"{x:+.4f}"))
    print(f"\n저장 완료 -> {RESULTS_DIR}/val_metrics.csv, val_monthly_ic.csv, importance.csv")


# ===========================================================================
# 단계 3: walk-forward CV
# ===========================================================================

def stage_cv(df: pd.DataFrame) -> None:
    header(f"[cv] walk-forward CV (val 연도 {CV_YEARS[0]}~{CV_YEARS[-1]})")
    rows = []
    for year in tqdm(CV_YEARS, desc="cv fold"):
        tr = df[(df["month"] >= SPLITS["train"][0]) & (df["month"] <= f"{year - 1}-12")]
        te = df[df["month"].str.startswith(str(year))]
        preds = predict_all(fit_all(tr, verbose=False), te)
        for name in MODEL_ORDER:
            ic = monthly_ic(te["month"], preds[name], te[TARGET])
            ls = monthly_long_short(te["month"], preds[name], te[TARGET])
            for month in ic.index:
                rows.append({"model": name, "month": month, "ic": ic[month], "ls": ls.get(month, np.nan)})
        fold = pd.DataFrame([r for r in rows if r["month"].startswith(str(year))])
        best = fold.groupby("model")["ic"].mean().sort_values(ascending=False)
        print(f"  {year}: " + ", ".join(f"{m}={v:.3f}" for m, v in best.head(3).items()))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "cv_monthly.csv", index=False)
    print(f"\n저장 완료 -> {RESULTS_DIR}/cv_monthly.csv")


# ===========================================================================
# 단계 4: 리포트 이미지
# ===========================================================================

def _heatmap(data: pd.DataFrame, path: Path, title: str, fmt: str, center_zero: bool) -> None:
    vals = data.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(0.75 * data.shape[1] + 3, 0.5 * data.shape[0] + 1.8))
    if center_zero:
        lim = np.nanmax(np.abs(vals)) or 1
        im = ax.imshow(vals, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    else:
        im = ax.imshow(vals, cmap="viridis", aspect="auto")
    ax.set_xticks(range(data.shape[1]), data.columns, rotation=60, ha="right")
    ax.set_yticks(range(data.shape[0]), data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, format(vals[i, j], fmt), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _comparison_table(summary: pd.DataFrame, path: Path) -> None:
    spec = [(c, label, fmt) for c, label, fmt in [
        ("val_ic_mean", "VAL IC", ".4f"), ("val_ic_ir", "VAL IC-IR", ".3f"),
        ("val_hit_rate", "VAL IC>0", ".0%"), ("val_ls_spread", "VAL L-S", ".2%"),
        ("cv_ic_mean", "CV IC", ".4f"), ("cv_ic_ir", "CV IC-IR", ".3f"),
        ("cv_hit_rate", "CV IC>0", ".0%"), ("cv_ls_spread", "CV L-S", ".2%"),
        ("cv_worst_year", "CV worst yr", ".4f"),
    ] if c in summary.columns]

    text, colors = [], []
    cmap = plt.get_cmap("RdYlGn")
    norm = {}
    for c, _, _ in spec:
        v = summary[c]
        norm[c] = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else v * 0 + 0.5
    for model in summary.index:
        text.append([format(summary.loc[model, c], f) for c, _, f in spec])
        colors.append([(*cmap(0.15 + 0.7 * norm[c][model])[:3], 0.6) for c, _, _ in spec])

    fig, ax = plt.subplots(figsize=(1.15 * len(spec) + 2.2, 0.42 * len(summary) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=text, cellColours=colors, rowLabels=list(summary.index),
                   colLabels=[label for _, label, _ in spec], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    sort_key = "CV IC" if "cv_ic_mean" in summary.columns else "VAL IC"
    ax.set_title(f"Model comparison (sorted by {sort_key}; L-S = top10% - bottom10% monthly return)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def stage_report() -> None:
    header("[report] 비교 이미지 생성")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(index=MODEL_ORDER)

    val_path = RESULTS_DIR / "val_metrics.csv"
    if val_path.exists():
        val = pd.read_csv(val_path).set_index("model")
        for c in ("ic_mean", "ic_ir", "hit_rate", "ls_spread"):
            summary[f"val_{c}"] = val[c]

    cv_path = RESULTS_DIR / "cv_monthly.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path, dtype={"month": str})
        for model, g in cv.groupby("model"):
            s = summarize(g["ic"], g["ls"].dropna())
            for c in ("ic_mean", "ic_ir", "hit_rate", "ls_spread"):
                summary.loc[model, f"cv_{c}"] = s[c]
        cv["year"] = cv["month"].str[:4]
        yearly = cv.groupby(["model", "year"])["ic"].mean().unstack().reindex(MODEL_ORDER)
        summary["cv_worst_year"] = yearly.min(axis=1)
        _heatmap(yearly, IMAGES_DIR / "cv_yearly_ic.png",
                 "Walk-forward CV: mean Rank IC by year", ".3f", center_zero=True)

    summary = summary.dropna(how="all")
    if summary.empty:
        print("results/에 결과가 없음 — val 또는 cv 단계를 먼저 실행")
        return
    sort_col = "cv_ic_mean" if "cv_ic_mean" in summary.columns else "val_ic_mean"
    summary = summary.sort_values(sort_col, ascending=False)
    summary.to_csv(RESULTS_DIR / "summary.csv")
    _comparison_table(summary, IMAGES_DIR / "model_comparison.png")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    imp_path = RESULTS_DIR / "importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        order = [m for m in summary.index if m in set(imp["model"])]
        feat = imp[imp["kind"] == "feature"].pivot(index="model", columns="name", values="delta_ic")
        feat = feat.loc[order, feat.mean().sort_values(ascending=False).index]
        _heatmap(feat, IMAGES_DIR / "importance_feature.png",
                 "Permutation importance on VAL: IC drop when feature is shuffled", ".3f", center_zero=True)
        grp = imp[imp["kind"] == "group"].pivot(index="model", columns="name", values="delta_ic")
        grp = grp.loc[order, grp.mean().sort_values(ascending=False).index]
        _heatmap(grp, IMAGES_DIR / "importance_group.png",
                 "Permutation importance on VAL: IC drop when group is shuffled", ".3f", center_zero=True)

    print(f"\n이미지 저장 완료 -> {IMAGES_DIR}")


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    stage_features()
    df = load_panel()
    stage_val(df)
    stage_cv(df)
    stage_report()


if __name__ == "__main__":
    main()
