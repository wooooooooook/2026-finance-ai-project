"""
news_cluster_pool.py: 012 실험 — 008/009의 attention pool은 "이 기사가 중요한지"를 기사
단위 점수로 지도학습해서 가중합했다. 이 실험은 그 가중치 학습 없이, 기사를 의미 임베딩
(기본 SBERT, 010에서 더 나은 설정을 찾으면 SEMANTIC_MODEL만 바꾸면 됨)으로 K=16개 클러스터로
나눈 뒤 "그 달 기사들이 클러스터에 분포된 비율"로 각 클러스터 중심(centroid)을 가중합한다
(pooled = Σ_c 비율_c x centroid_c). attention-pool의 "점수 MLP + softmax" 자리를 비지도
클러스터 비율로 완전히 대체한 버전 — 005 step9(attention 없는 단순 평균)의 클러스터 버전이라고
볼 수 있음. 그 pooled 벡터를 008/009와 똑같이 선형 projection(→8차원)해서 다음 달 평균
수익률을 예측하도록 학습시키고, 그 8차원을 최종 뉴스 feature로 쓴다.

평가는 009와 완전히 같은 틀(베이스라인 freeze + base_rank + 잔차보정 + placebo 대조군)을
그대로 쓴다 — 이유는 009 README 참고.

008/009와 다른 점 (비자명한 결정):
- "점수 MLP로 기사별 중요도를 배우는 attention" 대신, pooled 벡터 자체가 이미 "클러스터
  비율 x centroid"로 고정돼 있어서 학습 대상이 projection(+head)뿐이다 — 기사 임베딩을 매
  epoch 순회할 필요가 없어서(월별로 이미 하나의 고정 벡터) 009의 attention pool보다 학습이
  훨씬 가볍다.
- KMeans는 split/fold가 바뀔 때마다 그 시점까지의 train 기사로만 다시 학습(fit)하고, 전체
  기사에 클러스터를 할당(predict)한다 — attention pool 재학습과 같은 유출 방지 원칙.
- 감성분류(FinBERT)는 전혀 안 쓴다 — 대표값이 임베딩 그 자체라 011과 달리 감성 점수를
  따로 계산할 필요가 없음(의미 임베딩 1회만 돌리면 됨).

사용법:
    cd members/wooooooooook/012_news_cluster_pool
    python3 news_cluster_pool.py
"""

import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.cluster import KMeans
from tqdm import tqdm

# ===========================================================================
# 설정
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
RAW = PROJECT_ROOT / "raw"

FEATURES_PATH = BASE_DIR / "features" / "panel.csv.gz"
NEWS_CACHE_DIR = BASE_DIR / "news_cache"
SAMPLE_PATH = NEWS_CACHE_DIR / "samples.csv.gz"
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
BASE_FEATURES = [f for cols in FEATURE_GROUPS.values() for f in cols]

NEWS_PER_MONTH = 500
NEWS_PER_DAY_CAP = 300
NEWS_BODY_CHARS = 200

SEMANTIC_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

CLUSTER_K = 16          # 클러스터 수 (pooling 비율 계산용, feature 차원이 아님)
OUT_DIM = 8             # 최종 projection 차원 — 008/009 attention pool과 비교하기 위해 동일하게 맞춤
CLPOOL_COLS = [f"clpool{i}" for i in range(1, OUT_DIM + 1)]
PLACEBO_COLS = [f"placebo{i}" for i in range(1, OUT_DIM + 1)]
MONTHLY_COLS = set(MACRO_COLS + CLPOOL_COLS + PLACEBO_COLS)

PROJ_LR = 1e-3
PROJ_MAX_EPOCHS = 30

TARGET = "target_next"
TARGET_RANK = "target_rank"

N_CORES = os.cpu_count() or 1
SEED = 42
N_TREES, DEPTH, LR = 500, 4, 0.05

BASE_MODEL_ORDER = [
    "ridge", "mlp", "random_forest", "extra_trees",
    "xgb_reg", "xgb_rank_target", "xgb_pairwise", "lgbm_reg", "catboost_reg",
]
RESID_MODEL_ORDER = [
    "ridge", "mlp", "random_forest", "extra_trees", "xgb", "xgb_pairwise", "lgbm_reg", "catboost_reg",
]


# ===========================================================================
# 공용 함수
# ===========================================================================

def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def select_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    start, end = SPLITS[split]
    return df[(df["month"] >= start) & (df["month"] <= end)]


def rank_pct(values, months) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).groupby(np.asarray(months)).rank(pct=True).to_numpy()


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


# ===========================================================================
# 1. 재무·거래 feature 패널 (008/009와 동일 — 독립 복사)
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
    panel = panel[["month", "code"] + BASE_FEATURES + [TARGET]]

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(FEATURES_PATH, index=False)
    print(f"저장 완료 -> {FEATURES_PATH}, 행수={len(panel)}")


# ===========================================================================
# 2. 뉴스 월별 샘플링 (008/009와 동일 로직)
# ===========================================================================

def _read_news_day(path_str: str):
    path = Path(path_str)
    month = path.stem.replace("bigkinds_", "")[:7]
    d = pd.read_csv(path, usecols=["제목", "본문", "분석제외 여부"], dtype=str,
                    keep_default_na=False, encoding="utf-8-sig")
    d = d[(d["제목"].str.strip() != "") & (d["본문"].str.strip() != "")]
    d = d[d["분석제외 여부"].str.strip() == ""]
    d = d.drop_duplicates(["제목", "본문"])
    vc = d["제목"].value_counts()
    bad = vc[(vc >= 10) & (vc >= 0.01 * len(d))].index
    d = d[~d["제목"].isin(bad)]
    n = len(d)
    if n > NEWS_PER_DAY_CAP:
        d = d.sample(NEWS_PER_DAY_CAP, random_state=zlib.crc32(path.name.encode()) % (2 ** 32))
    text = (d["제목"].str.strip() + ". "
            + d["본문"].str.slice(0, NEWS_BODY_CHARS).str.replace(r"\s+", " ", regex=True))
    return month, n, text.tolist()


def _sample_month(days: list, rng) -> list[str]:
    texts, weights = [], []
    for n, t in days:
        if t:
            texts.extend(t)
            weights.extend([n / len(t)] * len(t))
    if not texts:
        return []
    p = np.asarray(weights) / np.sum(weights)
    idx = rng.choice(len(texts), size=min(NEWS_PER_MONTH, len(texts)), replace=False, p=p)
    return [texts[i] for i in sorted(idx)]


def stage_news_sample() -> None:
    header("[2] 뉴스 월별 샘플링")
    if SAMPLE_PATH.exists():
        print(f"이미 존재 -> {SAMPLE_PATH}")
        return

    files = sorted(str(p) for p in (RAW / "news").glob("bigkinds_*.csv.gz"))
    print(f"일별 파일 {len(files)}개")
    rng = np.random.default_rng(SEED)
    rows = []
    cur_month, days = None, []

    def flush():
        if cur_month is None:
            return
        for t in _sample_month(days, rng):
            rows.append((cur_month, t))

    with ProcessPoolExecutor(max_workers=N_CORES) as pool:
        for month, n, texts in tqdm(pool.map(_read_news_day, files, chunksize=8),
                                    total=len(files), desc="news files"):
            if month != cur_month:
                flush()
                cur_month, days = month, []
            days.append((n, texts))
        flush()

    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["month", "text"]).to_csv(SAMPLE_PATH, index=False)
    print(f"저장 완료 -> {SAMPLE_PATH}, {len(rows)}건")


# ===========================================================================
# 3. 의미 임베딩 (클러스터링용, SentenceTransformer) — 연도별 캐시
# ===========================================================================

def stage_semantic_embed() -> tuple[np.ndarray, np.ndarray]:
    header("[3] 의미 임베딩 (SentenceTransformer)")
    samples = pd.read_csv(SAMPLE_PATH, dtype={"month": str})
    years = sorted(samples["month"].str[:4].unique())
    todo = [y for y in years if not (NEWS_CACHE_DIR / f"semantic_{y}.npz").exists()]
    print(f"연도 {len(years)}개 중 임베딩 필요 {len(todo)}개")

    if todo:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SEMANTIC_MODEL, device="cpu")
        for year in todo:
            part = samples[samples["month"].str.startswith(year)]
            texts = part["text"].fillna("").astype(str).tolist()
            t0 = time.time()
            emb = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
            np.savez(NEWS_CACHE_DIR / f"semantic_{year}.npz",
                    month=part["month"].to_numpy().astype(str), embeddings=emb.astype(np.float32))
            print(f"  {year}: {len(part)}건, {time.time() - t0:.0f}s")

    months_all, emb_all = [], []
    for y in years:
        z = np.load(NEWS_CACHE_DIR / f"semantic_{y}.npz", allow_pickle=True)
        months_all.append(z["month"].astype(str))
        emb_all.append(z["embeddings"])
    return np.concatenate(months_all), np.concatenate(emb_all, axis=0)


# ===========================================================================
# 4. 클러스터 비율 pooling + projection 학습
#    pooled_그달 = Σ_c (그 달 기사 중 cluster c 비율) x centroid_c  (고정, 비지도)
#    projection(→OUT_DIM) + head(→1)만 "다음 달 평균 수익률 예측"으로 학습한다
#    (008/009 attention pool의 "점수 MLP"만 비지도 클러스터 비율로 대체한 구조)
# ===========================================================================

class ClusterProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = OUT_DIM):
        super().__init__()
        self.project = nn.Linear(in_dim, out_dim)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, pooled_vec: torch.Tensor):
        projected = self.project(pooled_vec)
        pred = self.head(projected)
        return projected, pred


def _epoch_loss(model, months, pooled_by_month, target_by_month, optimizer=None):
    total_loss, n = 0.0, 0
    for month in months:
        if month not in pooled_by_month or month not in target_by_month:
            continue
        pooled, targets = pooled_by_month[month], target_by_month[month]
        _, pred = model(pooled)
        loss = ((pred - targets) ** 2).mean()
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n if n else float("nan")


def fit_cluster_projector(pooled_by_month: dict, target_by_month: dict, train_months: list[str],
                          monitor_months: list[str], export_months: list[str],
                          in_dim: int, epochs: int | None = None, seed: int = SEED):
    assert monitor_months or epochs, "epochs 고정값이 없으면 monitor_months(조기종료 기준)가 필요함"
    torch.manual_seed(seed)
    model = ClusterProjector(in_dim=in_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=PROJ_LR)

    best_state, best_loss, best_epoch = None, float("inf"), epochs or PROJ_MAX_EPOCHS
    for epoch in range(epochs or PROJ_MAX_EPOCHS):
        model.train()
        _epoch_loss(model, train_months, pooled_by_month, target_by_month, optimizer)
        if monitor_months:
            model.eval()
            with torch.no_grad():
                val_loss = _epoch_loss(model, monitor_months, pooled_by_month, target_by_month)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    rows, idx = [], []
    with torch.no_grad():
        for m in export_months:
            if m in pooled_by_month:
                projected, _ = model(pooled_by_month[m])
                rows.append(projected.numpy())
                idx.append(m)
    feat = pd.DataFrame(rows, index=idx, columns=CLPOOL_COLS)
    return feat, best_epoch, (best_loss if monitor_months else float("nan"))


def build_cluster_pool_features(df: pd.DataFrame, pool_month: np.ndarray, semantic_emb: np.ndarray,
                                train_end: str, monitor_months: list[str] | None = None,
                                epochs: int | None = None, k: int = CLUSTER_K, seed: int = SEED):
    train_mask = (pool_month >= SPLITS["train"][0]) & (pool_month <= train_end)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(semantic_emb[train_mask])
    labels = km.predict(semantic_emb)
    centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)  # (k, embed_dim)

    props = (pd.DataFrame({"month": pool_month, "cluster": labels})
               .groupby(["month", "cluster"]).size().unstack("cluster", fill_value=0))
    props = props.reindex(columns=range(k), fill_value=0)
    props = props.div(props.sum(axis=1), axis=0).fillna(0.0)

    pooled_by_month = {
        m: torch.tensor(props.loc[m].to_numpy(), dtype=torch.float32) @ centroids
        for m in props.index
    }
    target_by_month = {
        m: torch.tensor(g[TARGET].to_numpy(), dtype=torch.float32)
        for m, g in df.groupby("month") if m in pooled_by_month
    }
    train_months = [m for m in pooled_by_month if SPLITS["train"][0] <= m <= train_end]
    export_months = sorted(pooled_by_month)
    return fit_cluster_projector(pooled_by_month, target_by_month, train_months, monitor_months or [],
                                 export_months, in_dim=centroids.shape[1], epochs=epochs)


def build_placebo_features(months: list[str], k: int = OUT_DIM, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(months), k))
    return pd.DataFrame(data, index=months, columns=PLACEBO_COLS)


# ===========================================================================
# 모델 zoo (009/011과 동일)
# ===========================================================================

class RankScaler:
    def __init__(self, features: list[str]):
        self.stock = [f for f in features if f not in MONTHLY_COLS]
        self.monthly = [f for f in features if f in MONTHLY_COLS]

    def fit(self, df: pd.DataFrame) -> "RankScaler":
        self.mu = df[self.monthly].mean()
        self.sd = df[self.monthly].std().replace(0, 1)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        stock = (df[self.stock].groupby(df["month"]).rank(pct=True) - 0.5).fillna(0.0)
        monthly = ((df[self.monthly] - self.mu) / self.sd).fillna(0.0)
        return np.hstack([stock.to_numpy(), monthly.to_numpy()])


class Model:
    def __init__(self, est, features: list[str], target: str,
                 rank_input: bool = False, impute: bool = False):
        self.est, self.features, self.target = est, features, target
        self.rank_input, self.impute = rank_input, impute

    def _X(self, df: pd.DataFrame, fit: bool = False):
        if self.rank_input:
            if fit:
                self.scaler = RankScaler(self.features).fit(df)
            return self.scaler.transform(df)
        X = df[self.features]
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
    def __init__(self, features: list[str], target: str):
        self.features, self.target = features, target

    def fit(self, df: pd.DataFrame) -> "XGBPairwise":
        d = df.sort_values("month", kind="stable")
        pct = d.groupby("month")[self.target].rank(pct=True).to_numpy()
        label = np.minimum((pct * 10).astype(int), 9)
        qid = pd.factorize(d["month"])[0]
        self.est = xgb.XGBRanker(
            objective="rank:pairwise",
            lambdarank_pair_method="mean",
            lambdarank_num_pair_per_sample=8,
            n_estimators=N_TREES, max_depth=DEPTH, learning_rate=LR,
            random_state=SEED, n_jobs=N_CORES, missing=np.nan,
        )
        self.est.fit(d[self.features], label, qid=qid)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.est.predict(df[self.features]), dtype=float)


def make_base_models(features: list[str]) -> dict:
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
        "ridge": Model(Ridge(alpha=1.0), features, target=TARGET_RANK, rank_input=True),
        "mlp": Model(
            MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=1024,
                         learning_rate_init=1e-3, max_iter=100, early_stopping=True,
                         n_iter_no_change=10, random_state=SEED),
            features, target=TARGET_RANK, rank_input=True,
        ),
        "random_forest": Model(RandomForestRegressor(**forest_kw), features, target=TARGET, impute=True),
        "extra_trees": Model(ExtraTreesRegressor(**forest_kw), features, target=TARGET, impute=True),
        "xgb_reg": Model(xgb.XGBRegressor(**xgb_kw), features, target=TARGET),
        "xgb_rank_target": Model(xgb.XGBRegressor(**xgb_kw), features, target=TARGET_RANK),
        "xgb_pairwise": XGBPairwise(features, target=TARGET),
        "lgbm_reg": Model(lgb.LGBMRegressor(
            n_estimators=N_TREES, max_depth=DEPTH, num_leaves=2 ** DEPTH - 1, learning_rate=LR,
            min_child_samples=100, random_state=SEED, n_jobs=N_CORES, verbose=-1,
        ), features, target=TARGET),
        "catboost_reg": Model(CatBoostRegressor(
            iterations=N_TREES, depth=DEPTH, learning_rate=LR,
            random_seed=SEED, thread_count=N_CORES, verbose=0,
        ), features, target=TARGET),
    }


def make_resid_models(features: list[str]) -> dict:
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
        "ridge": Model(Ridge(alpha=1.0), features, target="resid", rank_input=True),
        "mlp": Model(
            MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=1024,
                         learning_rate_init=1e-3, max_iter=100, early_stopping=True,
                         n_iter_no_change=10, random_state=SEED),
            features, target="resid", rank_input=True,
        ),
        "random_forest": Model(RandomForestRegressor(**forest_kw), features, target="resid", impute=True),
        "extra_trees": Model(ExtraTreesRegressor(**forest_kw), features, target="resid", impute=True),
        "xgb": Model(xgb.XGBRegressor(**xgb_kw), features, target="resid"),
        "xgb_pairwise": XGBPairwise(features, target="resid"),
        "lgbm_reg": Model(lgb.LGBMRegressor(
            n_estimators=N_TREES, max_depth=DEPTH, num_leaves=2 ** DEPTH - 1, learning_rate=LR,
            min_child_samples=100, random_state=SEED, n_jobs=N_CORES, verbose=-1,
        ), features, target="resid"),
        "catboost_reg": Model(CatBoostRegressor(
            iterations=N_TREES, depth=DEPTH, learning_rate=LR,
            random_seed=SEED, thread_count=N_CORES, verbose=0,
        ), features, target="resid"),
    }


def fit_models(models: dict, df: pd.DataFrame, verbose: bool = True, label: str = "") -> dict:
    out = {}
    for name, m in models.items():
        t0 = time.time()
        out[name] = m.fit(df)
        if verbose:
            tag = f"[{label}] " if label else ""
            print(f"  {tag}{name:16s} 학습 완료 ({time.time() - t0:.1f}s)")
    return out


def ensemble_pred(preds: dict, months) -> np.ndarray:
    months = np.asarray(months)
    ranks = [pd.Series(p).groupby(months).rank(pct=True).to_numpy() for p in preds.values()]
    return np.mean(ranks, axis=0)


# ===========================================================================
# 핵심: 베이스라인 freeze + 클러스터pool/placebo 잔차보정 증분 평가 (009와 동일 로직)
# ===========================================================================

def evaluate_incremental(train_df: pd.DataFrame, eval_df: pd.DataFrame, verbose: bool = True) -> list[dict]:
    base_fits = fit_models(make_base_models(BASE_FEATURES), train_df, verbose=verbose, label="baseline")
    base_pred_train = {m: mod.predict(train_df) for m, mod in base_fits.items()}
    base_pred_eval = {m: mod.predict(eval_df) for m, mod in base_fits.items()}
    base_pred_train["ensemble"] = ensemble_pred(base_pred_train, train_df["month"])
    base_pred_eval["ensemble"] = ensemble_pred(base_pred_eval, eval_df["month"])

    target_rank_train = rank_pct(train_df[TARGET].to_numpy(), train_df["month"])
    records = []

    baseline_names = BASE_MODEL_ORDER + ["ensemble"]
    bm_iter = tqdm(baseline_names, desc="baseline", leave=False) if verbose else baseline_names
    for bm in bm_iter:
        base_rank_tr = rank_pct(base_pred_train[bm], train_df["month"])
        base_rank_ev = rank_pct(base_pred_eval[bm], eval_df["month"])

        ic = monthly_ic(eval_df["month"], base_rank_ev, eval_df[TARGET])
        records += [{"baseline": bm, "resid_model": "(base only)", "variant": "base", "month": mo, "ic": v}
                    for mo, v in ic.items()]

        resid_tr = target_rank_train - base_rank_tr
        tr_aux = pd.DataFrame({"month": train_df["month"].to_numpy(), "base_rank": base_rank_tr, "resid": resid_tr})
        ev_aux = pd.DataFrame({"month": eval_df["month"].to_numpy(), "base_rank": base_rank_ev})

        for variant, cols in (("clpool", CLPOOL_COLS), ("placebo", PLACEBO_COLS)):
            tr_v = pd.concat([tr_aux, train_df[cols].reset_index(drop=True)], axis=1)
            ev_v = pd.concat([ev_aux, eval_df[cols].reset_index(drop=True)], axis=1)
            feat_cols = ["base_rank"] + cols

            resid_fits = fit_models(make_resid_models(feat_cols), tr_v, verbose=False, label=f"{bm}/{variant}")
            combined_by_arch = {}
            for arch, mod in resid_fits.items():
                combined = base_rank_ev + mod.predict(ev_v)
                combined_by_arch[arch] = combined
                ic = monthly_ic(eval_df["month"], combined, eval_df[TARGET])
                records += [{"baseline": bm, "resid_model": arch, "variant": variant, "month": mo, "ic": v}
                            for mo, v in ic.items()]

            ens_combined = ensemble_pred(combined_by_arch, eval_df["month"])
            ic = monthly_ic(eval_df["month"], ens_combined, eval_df[TARGET])
            records += [{"baseline": bm, "resid_model": "ensemble", "variant": variant, "month": mo, "ic": v}
                        for mo, v in ic.items()]

    return records


def summarize_incremental(ic_df: pd.DataFrame) -> pd.DataFrame:
    g = ic_df.groupby(["baseline", "resid_model", "variant"])["ic"].mean().reset_index()
    base_ic = g[g["variant"] == "base"].set_index("baseline")["ic"]
    out = pd.DataFrame(index=base_ic.index)
    out["base_ic"] = base_ic
    for variant in ("clpool", "placebo"):
        sub = g[g["variant"] == variant]
        ens = sub[sub["resid_model"] == "ensemble"].set_index("baseline")["ic"]
        best_idx = sub.loc[sub.groupby("baseline")["ic"].idxmax()].set_index("baseline")
        out[f"{variant}_ensemble_ic"] = ens
        out[f"{variant}_ensemble_delta"] = ens - out["base_ic"]
        out[f"{variant}_best_ic"] = best_idx["ic"]
        out[f"{variant}_best_arch"] = best_idx["resid_model"]
        out[f"{variant}_best_delta"] = best_idx["ic"] - out["base_ic"]
    out["clpool_vs_placebo_ensemble"] = out["clpool_ensemble_delta"] - out["placebo_ensemble_delta"]
    out["clpool_vs_placebo_best"] = out["clpool_best_delta"] - out["placebo_best_delta"]
    order = [m for m in BASE_MODEL_ORDER + ["ensemble"] if m in out.index]
    return out.loc[order]


def stage_incremental_val(df: pd.DataFrame, pool_month: np.ndarray, semantic_emb: np.ndarray) -> int:
    header("[4] 클러스터 pooling projection 학습 + 베이스라인 freeze + 잔차보정 증분 평가 (VAL)")
    months_all = sorted(df["month"].unique())
    monitor_months = [m for m in months_all if SPLITS["val"][0] <= m <= SPLITS["val"][1]]
    clpool_feat, best_epoch, best_loss = build_cluster_pool_features(
        df, pool_month, semantic_emb, train_end=SPLITS["train"][1], monitor_months=monitor_months,
    )
    print(f"projector: best_epoch={best_epoch}/{PROJ_MAX_EPOCHS}, val_loss={best_loss:.6f}")

    placebo_feat = build_placebo_features(months_all)
    dfn = (df.merge(clpool_feat, left_on="month", right_index=True, how="left")
             .merge(placebo_feat, left_on="month", right_index=True, how="left"))
    train_df, val_df = select_split(dfn, "train"), select_split(dfn, "val")
    print(f"train={len(train_df)}, val={len(val_df)}")

    records = evaluate_incremental(train_df, val_df, verbose=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ic_df = pd.DataFrame(records)
    ic_df.to_csv(RESULTS_DIR / "val_incremental_ic.csv", index=False)

    summary = summarize_incremental(ic_df)
    summary.to_csv(RESULTS_DIR / "val_incremental_summary.csv")
    print("\n" + summary.to_string(float_format=lambda x: f"{x:+.4f}" if isinstance(x, float) else str(x)))
    return best_epoch


def stage_incremental_cv(df: pd.DataFrame, pool_month: np.ndarray, semantic_emb: np.ndarray,
                         proj_epochs: int) -> None:
    header(f"[5] walk-forward CV (val 연도 {CV_YEARS[0]}~{CV_YEARS[-1]}, "
           f"베이스라인 freeze + 클러스터pool 잔차보정, projector epoch={proj_epochs} 고정)")
    months_all = sorted(df["month"].unique())
    placebo_feat_full = build_placebo_features(months_all)
    all_records = []
    for year in tqdm(CV_YEARS, desc="cv fold"):
        clpool_feat, _, _ = build_cluster_pool_features(
            df, pool_month, semantic_emb, train_end=f"{year - 1}-12", epochs=proj_epochs,
        )
        dfn = (df.merge(clpool_feat, left_on="month", right_index=True, how="left")
                 .merge(placebo_feat_full, left_on="month", right_index=True, how="left"))
        tr = dfn[(dfn["month"] >= SPLITS["train"][0]) & (dfn["month"] <= f"{year - 1}-12")]
        te = dfn[dfn["month"].str.startswith(str(year))]

        recs = evaluate_incremental(tr, te, verbose=False)
        for r in recs:
            r["year"] = year
        all_records += recs

        def mean_ic(baseline, resid_model, variant):
            vals = [r["ic"] for r in recs
                    if r["baseline"] == baseline and r["resid_model"] == resid_model and r["variant"] == variant]
            return float(np.mean(vals)) if vals else float("nan")

        base_ic = mean_ic("ensemble", "(base only)", "base")
        cl_ic = mean_ic("ensemble", "ensemble", "clpool")
        pl_ic = mean_ic("ensemble", "ensemble", "placebo")
        print(f"  {year}: base(ensemble)={base_ic:.3f}, "
              f"+clpool(ensemble resid)={cl_ic:.3f} (Δ{cl_ic - base_ic:+.3f}), "
              f"+placebo={pl_ic:.3f} (Δ{pl_ic - base_ic:+.3f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_df = pd.DataFrame(all_records)
    cv_df.to_csv(RESULTS_DIR / "cv_incremental_ic.csv", index=False)

    summary = summarize_incremental(cv_df)
    summary.to_csv(RESULTS_DIR / "cv_incremental_summary.csv")
    print("\n" + summary.to_string(float_format=lambda x: f"{x:+.4f}" if isinstance(x, float) else str(x)))


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    stage_features()
    stage_news_sample()
    pool_month, semantic_emb = stage_semantic_embed()

    df = pd.read_csv(FEATURES_PATH, dtype={"month": str, "code": str})
    df[TARGET_RANK] = df.groupby("month")[TARGET].rank(pct=True) - 0.5
    best_epoch = stage_incremental_val(df, pool_month, semantic_emb)
    stage_incremental_cv(df, pool_month, semantic_emb, best_epoch)


if __name__ == "__main__":
    main()
