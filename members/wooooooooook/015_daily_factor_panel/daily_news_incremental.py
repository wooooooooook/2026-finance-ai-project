"""
daily_news_incremental.py: 015 4단계 — 009(news_incremental, 월별)와 같은 방식으로,
daily_factor_panel.py가 학습한 베이스라인(23-factor)을 완전히 고정(freeze)한 채
014(news_horizon_decay)에서 만든 일별 뉴스 attention-pool 피처(8차원)가 "베이스라인
예측 순위(base_rank)와의 상호작용"으로 실제 증분가치를 주는지 검증한다.

핵심 설계 (009/014와 동일한 논리를 일별로 적용):
  - raw/news는 기업별이 아니라 경제일간지 전체라, 뉴스 피처는 그 날 전 종목에 똑같은
    값이 붙는다. 그래서 뉴스 자체는 횡단면 순위(Rank IC)에 기여할 수 없고, base_rank
    (종목마다 다른 값)와의 상호작용을 통해서만 기여할 수 있다 — base_rank를 함께
    입력으로 주는 2단계(잔차) 모델을 쓰는 이유.
  - placebo 대조군: 뉴스 8차원 대신 같은 구조(날짜별로 하나, 그 날 전 종목에 동일 값)의
    랜덤 노이즈 8차원을 넣고 똑같이 학습 — "뉴스라서 좋아진 것"과 "입력을 8개 더 줘서
    생기는 capacity 증가 효과"를 구분. 014와 동일하게 시드 20개 평균(단일 시드는 운일
    수 있음).
  - 2단계 모델은 009의 9-아키텍처 zoo 대신 XGBoost 하나만 사용(014와 같은 수준의 축소 —
    일별 데이터가 월별보다 20배 커서 9개 아키텍처 전부는 비용이 너무 큼).

005/008/009 관례대로 이 파일은 daily_factor_panel.py를 import하지 않고 필요한 상수를
자체 복사한다. 대신 그 파일이 저장한 features/panel.csv.gz, models/xgb_all.json은 그대로
읽어서 재사용한다(같은 015 폴더 안의 결과물이라 "실험 폴더끼리 import 금지" 규칙과 무관).

전제: daily_factor_panel.py 실행 완료(panel.csv.gz, xgb_all.json 존재),
014_news_horizon_decay의 news_feat_daily_8.csv 존재.

사용법:
    cd members/wooooooooook/015_daily_factor_panel
    python3 daily_news_incremental.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

FEATURES_PATH = BASE_DIR / "features" / "panel.csv.gz"
MODEL_PATH = BASE_DIR / "models" / "xgb_all.json"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"
NEWS_FEAT_PATH = (
    PROJECT_ROOT / "members" / "wooooooooook" / "014_news_horizon_decay"
    / "features" / "news_feat_daily_8.csv"
)

# daily_factor_panel.py와 동일(복사) — 005/008/009 관례상 import 대신 자체 보유
SPLITS = {
    "train": ("2001-01-01", "2017-12-31"),
    "val": ("2018-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2025-12-31"),
}
FEATURE_GROUPS = {
    "return": ["ret_1m", "ret_3m", "mom_12_2", "vol_12m"],
    "size": ["log_mktcap", "log_price", "share_issuance_12m"],
    "liquidity": ["turnover_1m", "abnormal_turnover", "amihud_illiq"],
    "individual_flow": ["individual_net_intensity", "individual_trade_intensity"],
    "foreign_flow": ["foreign_net_intensity", "foreign_trade_intensity"],
    "foreign_holding": [
        "foreign_own_ratio", "foreign_own_chg_1m", "foreign_own_chg_3m", "foreign_limit_exhaustion",
    ],
    "macro": ["base_rate", "fx_change", "term_spread", "cpi_yoy", "kospi_rv"],
}
ALL_FEATURES = [f for cols in FEATURE_GROUPS.values() for f in cols]
TARGET = "target_next"

N_CORES = os.cpu_count() or 1
XGB_PARAMS = dict(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=42, missing=np.nan)
SEED = 42
N_PLACEBO_REPEATS = 20
NEWS_DIM = 8
NEWS_COLS = [f"news_daily_8_{i}" for i in range(NEWS_DIM)]
PLACEBO_COLS = [f"noise_{i}" for i in range(NEWS_DIM)]

MODEL_COLORS = {"base": "#2a78d6", "news": "#1baf7a", "placebo": "#9a9a95"}


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def select_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    start, end = SPLITS[split]
    return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def rank_pct(values, groups) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).groupby(np.asarray(groups)).rank(pct=True).to_numpy()


def daily_ic(dates, pred, actual) -> pd.Series:
    """일별 횡단면 Spearman 상관. groupby.apply의 Series/DataFrame 모양 애매함을 피하려고
    groupby를 직접 순회해서 dict -> Series로 고정한다(daily_factor_panel.py와 동일 패턴)."""
    tmp = pd.DataFrame({
        "date": np.asarray(dates),
        "pred": np.asarray(pred, dtype=float),
        "actual": np.asarray(actual, dtype=float),
    }).dropna()
    tmp["pred_r"] = tmp.groupby("date")["pred"].rank()
    tmp["actual_r"] = tmp.groupby("date")["actual"].rank()
    corr_by_date = {
        date: g["pred_r"].corr(g["actual_r"]) if len(g) >= 10 else np.nan
        for date, g in tmp.groupby("date")
    }
    return pd.Series(corr_by_date).dropna()


def ic_summary(ic: pd.Series) -> dict:
    std = ic.std()
    return {
        "ic_mean": ic.mean(),
        "ic_std": std,
        "ic_ir": ic.mean() / std if std else np.nan,
        "hit_rate": (ic > 0).mean(),
        "n_days": len(ic),
    }


def format_summary(s: dict) -> str:
    return f"IC={s['ic_mean']:.4f}  IC-IR={s['ic_ir']:.4f}  양의 IC 비율={s['hit_rate']:.1%} ({s['n_days']}일)"


def load_panel() -> pd.DataFrame:
    return pd.read_csv(FEATURES_PATH, dtype={"date": str, "code": str})


def load_baseline_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(n_jobs=N_CORES)
    model.load_model(MODEL_PATH)
    return model


def load_news_features() -> pd.DataFrame:
    news = pd.read_csv(NEWS_FEAT_PATH, dtype={"key": str})
    news = news.rename(columns={"key": "date"}).set_index("date")
    return news[NEWS_COLS]


def broadcast_values(dates: pd.Series, feat_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """feat_df: index=date, columns=cols인 날짜별 값을 dates(종목×날짜 긴 형태)에 맞춰
    그 날 전 종목에 동일 값으로 펼친다(뉴스·placebo 둘 다 이 구조)."""
    return pd.DataFrame({c: dates.map(feat_df[c]).to_numpy() for c in cols})


def make_placebo(dates: pd.Series, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    uniq = np.sort(dates.unique())
    noise = rng.standard_normal((len(uniq), NEWS_DIM))
    return pd.DataFrame(noise, index=uniq, columns=PLACEBO_COLS)


def fit_resid_xgb(X: pd.DataFrame, y: np.ndarray) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=N_CORES)
    model.fit(X, y)
    return model


def run_variant(
    base_rank_train: np.ndarray, base_rank_val: np.ndarray, resid_train: np.ndarray,
    block_train: pd.DataFrame, block_val: pd.DataFrame, val_dates: pd.Series, val_target: pd.Series,
) -> pd.Series:
    """[base_rank] + block(뉴스 또는 placebo 8차원) -> resid 학습, val에서
    combined = base_rank + 예측값으로 다시 합쳐 IC를 계산한다(009의 2단계 stacking)."""
    cols = list(block_train.columns)
    feat_cols = ["base_rank"] + cols
    tr = pd.concat([pd.DataFrame({"base_rank": base_rank_train}), block_train], axis=1)
    model = fit_resid_xgb(tr[feat_cols], resid_train)

    ev = pd.concat([pd.DataFrame({"base_rank": base_rank_val}), block_val], axis=1)
    combined = base_rank_val + model.predict(ev[feat_cols])
    return daily_ic(val_dates, combined, val_target)


def main() -> None:
    header("[015-4] 베이스라인 freeze + 뉴스/placebo 잔차보정 (일별)")
    df = load_panel()
    train_df = select_split(df, "train")
    val_df = select_split(df, "val")
    print(f"train 행수={len(train_df)}, val 행수={len(val_df)}")

    model = load_baseline_model()
    base_pred_train = model.predict(train_df[ALL_FEATURES])
    base_pred_val = model.predict(val_df[ALL_FEATURES])

    base_rank_train = rank_pct(base_pred_train, train_df["date"])
    base_rank_val = rank_pct(base_pred_val, val_df["date"])
    target_rank_train = rank_pct(train_df[TARGET].to_numpy(), train_df["date"])
    resid_train = target_rank_train - base_rank_train

    base_ic = daily_ic(val_df["date"], base_rank_val, val_df[TARGET])
    base_summary = ic_summary(base_ic)
    print(f"\n[base] {format_summary(base_summary)}")

    news = load_news_features()
    news_block_train = broadcast_values(train_df["date"], news, NEWS_COLS)
    news_block_val = broadcast_values(val_df["date"], news, NEWS_COLS)
    news_ic = run_variant(
        base_rank_train, base_rank_val, resid_train,
        news_block_train, news_block_val, val_df["date"], val_df[TARGET],
    )
    news_summary = ic_summary(news_ic)
    print(f"[+news] {format_summary(news_summary)}  (delta vs base: {news_summary['ic_mean'] - base_summary['ic_mean']:+.4f})")

    placebo_means = []
    for rep in tqdm(range(N_PLACEBO_REPEATS), desc="placebo"):
        noise = make_placebo(pd.concat([train_df["date"], val_df["date"]]), SEED + rep)
        placebo_block_train = broadcast_values(train_df["date"], noise, PLACEBO_COLS)
        placebo_block_val = broadcast_values(val_df["date"], noise, PLACEBO_COLS)
        rep_ic = run_variant(
            base_rank_train, base_rank_val, resid_train,
            placebo_block_train, placebo_block_val, val_df["date"], val_df[TARGET],
        )
        placebo_means.append(ic_summary(rep_ic)["ic_mean"])
    placebo_means = np.array(placebo_means)
    placebo_mean, placebo_std = placebo_means.mean(), placebo_means.std()
    print(f"[+placebo] {N_PLACEBO_REPEATS}회 평균 IC={placebo_mean:.4f} +- {placebo_std:.4f} "
          f"(min {placebo_means.min():.4f}, max {placebo_means.max():.4f})")
    print(f"\n핵심 숫자 — news_delta - placebo_delta: "
          f"{(news_summary['ic_mean'] - base_summary['ic_mean']) - (placebo_mean - base_summary['ic_mean']):+.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([
        {"model": "base", "ic_mean": base_summary["ic_mean"], "ic_std": base_summary["ic_std"]},
        {"model": "news", "ic_mean": news_summary["ic_mean"], "ic_std": news_summary["ic_std"]},
        {"model": "placebo_mean", "ic_mean": placebo_mean, "ic_std": placebo_std},
    ])
    summary.to_csv(RESULTS_DIR / "news_incremental_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["base", "news", "placebo_mean"]
    means = [base_summary["ic_mean"], news_summary["ic_mean"], placebo_mean]
    errs = [0, 0, placebo_std]
    ax.bar(names, means, yerr=errs, color=[MODEL_COLORS.get(n.split("_")[0], "#888") for n in names])
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("VAL Rank IC (일별)")
    ax.set_title("daily baseline vs +news vs +placebo")
    fig.tight_layout()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES_DIR / "news_incremental.png", dpi=150)
    plt.close(fig)
    print(f"\n저장 완료 -> {RESULTS_DIR / 'news_incremental_summary.csv'}, {IMAGES_DIR / 'news_incremental.png'}")


if __name__ == "__main__":
    main()
