"""
news_cluster_sentiment.py: 011 실험 — 008/009의 뉴스 feature는 "경제일간지 전체 기사를
attention pool로 뭉쳐 8차원 하나"였다. 이 실험은 기사를 의미 임베딩(기본 SBERT, 010에서
더 나은 모델/설정을 찾으면 EMBED_MODEL만 바꾸면 됨)으로 K=16개 클러스터로 나누고, 클러스터
별로 그 달 기사들의 FinBERT 감성 점수를 평균 낸 "클러스터별 감성" 16차원을 뉴스 feature로
쓴다. 008의 8차원 attention-pool 벡터 하나보다 "토픽별로 쪼개진" 입력이 비선형 모델(트리)
입장에서 상호작용을 찾기 더 쉬울 수 있다는 가설.

009와 똑같은 freeze+잔차+placebo 평가 틀을 그대로 쓴다(복붙) — 008처럼 base 모델까지 통째로
다시 학습하면 VAL에서 좋아 보이다 CV에서 사라지는 과적합과 구분이 안 됨.
클러스터 감성도 뉴스와 똑같이 "그 달 전 종목에 같은 값"이라, base_rank(베이스라인의 그 달
예측 순위)를 2단계 모델에 같이 넣어야 상호작용을 통해서라도 기여할 수 있다 — 자세한 이유는
009 README 참고. placebo는 16차원 랜덤 노이즈로, "클러스터 16개라서" 생기는 capacity 증가
효과와 "감성 신호 자체"를 구분한다.

008/009와의 차이 (비자명한 결정):
- attention pool(지도학습, next-month-return으로 fit)이 없다 — 클러스터링은 비지도라서
  "몇 epoch 학습했는지" 같은 조기종료 로직 자체가 필요 없음. 그 대신 split/fold가 바뀔 때마다
  KMeans를 그 시점까지의 train 기사로만 다시 학습(fit)하고, 전체 기사에 대해 cluster를
  할당(predict)한다 — attention pool을 분할마다 재학습했던 것과 같은 유출 방지 원칙.
- FinBERT를 임베딩이 아니라 "감성분류 헤드 포함"으로 로드해서 긍정-부정 확률 차를 기사별
  점수로 쓴다(008/009는 분류 헤드를 버리고 임베딩만 썼음). 라벨 순서가 모델마다 달라서,
  id2label을 코드에서 읽어 positive/negative 인덱스를 찾는다(하드코딩 안 함) — 실행 로그에
  찾은 매핑이 그대로 출력되니 실행 후 한번 확인할 것.
- 의미 임베딩(클러스터링용)과 감성 점수(집계 대상)를 분리해서 두 번 모델을 돌려야 해서,
  009 하나보다 전처리 비용이 두 배 가깐다. 그래서 월 샘플을 1,000개가 아니라 500개로 줄였다
  (008/009와 정확히 같은 샘플은 아니지만, 이 실험의 목적은 "클러스터 감성이라는 입력 형태"
  자체가 도움이 되는지 보는 것이라 완전히 같은 샘플일 필요는 없음).

사용법:
    cd members/wooooooooook/011_news_cluster_sentiment
    python3 news_cluster_sentiment.py
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

# 뉴스 샘플링
NEWS_PER_MONTH = 500
NEWS_PER_DAY_CAP = 300
NEWS_BODY_CHARS = 200

# 의미 임베딩 (클러스터링용) — 010에서 더 나은 설정을 찾으면 여기만 바꾸면 됨
SEMANTIC_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
# 감성 점수 (집계 대상)
SENTIMENT_MODEL = "snunlp/KR-FinBert-SC"
EMBED_MAX_LEN = 256
EMBED_BATCH = 32

CLUSTER_K = 16
CLSENT_COLS = [f"clsent{i}" for i in range(1, CLUSTER_K + 1)]
PLACEBO_COLS = [f"placebo{i}" for i in range(1, CLUSTER_K + 1)]
MONTHLY_COLS = set(MACRO_COLS + CLSENT_COLS + PLACEBO_COLS)

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
# 3. 의미 임베딩(클러스터링용) + 감성 점수(집계 대상) — 연도별 캐시
# ===========================================================================

def stage_semantic_embed() -> tuple[np.ndarray, np.ndarray]:
    header("[3] 의미 임베딩 (클러스터링용, SentenceTransformer)")
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


def stage_sentiment_embed() -> np.ndarray:
    header("[4] 감성 점수 (FinBERT 분류 헤드, 기사 단위)")
    samples = pd.read_csv(SAMPLE_PATH, dtype={"month": str})
    years = sorted(samples["month"].str[:4].unique())
    todo = [y for y in years if not (NEWS_CACHE_DIR / f"sentiment_{y}.npz").exists()]
    print(f"연도 {len(years)}개 중 감성 계산 필요 {len(todo)}개")

    if todo:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(N_CORES)
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        pos_idx = [i for i, lab in id2label.items() if "pos" in lab or "긍" in lab]
        neg_idx = [i for i, lab in id2label.items() if "neg" in lab or "부" in lab]
        assert len(pos_idx) == 1 and len(neg_idx) == 1, f"감성 라벨 매핑을 확인하세요: {id2label}"
        pos_idx, neg_idx = pos_idx[0], neg_idx[0]
        print(f"감성 라벨 매핑: {id2label} -> positive idx={pos_idx}, negative idx={neg_idx}")

        for year in todo:
            part = samples[samples["month"].str.startswith(year)]
            texts = part["text"].fillna("").astype(str).tolist()
            t0 = time.time()
            scores = []
            for i in tqdm(range(0, len(texts), EMBED_BATCH), desc=f"sentiment {year}"):
                batch = texts[i:i + EMBED_BATCH]
                enc = tokenizer(batch, padding=True, truncation=True, max_length=EMBED_MAX_LEN, return_tensors="pt")
                with torch.no_grad():
                    probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
                scores.append(probs[:, pos_idx] - probs[:, neg_idx])
            score = np.concatenate(scores).astype(np.float32)
            np.savez(NEWS_CACHE_DIR / f"sentiment_{year}.npz",
                    month=part["month"].to_numpy().astype(str), score=score)
            print(f"  {year}: {len(part)}건, {time.time() - t0:.0f}s")

    scores_all = []
    for y in years:
        z = np.load(NEWS_CACHE_DIR / f"sentiment_{y}.npz", allow_pickle=True)
        scores_all.append(z["score"])
    return np.concatenate(scores_all)


def build_cluster_sentiment_features(pool_month: np.ndarray, semantic_emb: np.ndarray, sentiment: np.ndarray,
                                     train_end: str, export_months: list[str], k: int = CLUSTER_K,
                                     seed: int = SEED) -> pd.DataFrame:
    """train_end까지의 기사로만 KMeans(k)를 학습(fit)하고, 전체 기사에 cluster를 할당(predict)한다
    (attention pool을 분할마다 재학습하던 것과 같은 유출 방지 원칙). export_months 전체에 대해
    (month, cluster) 평균 감성 테이블을 만들어 반환한다."""
    train_mask = (pool_month >= SPLITS["train"][0]) & (pool_month <= train_end)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(semantic_emb[train_mask])
    labels = km.predict(semantic_emb)

    agg = (pd.DataFrame({"month": pool_month, "cluster": labels, "sent": sentiment})
             .groupby(["month", "cluster"])["sent"].mean().unstack("cluster"))
    agg = agg.reindex(index=export_months, columns=range(k))
    agg.columns = [f"clsent{i + 1}" for i in range(k)]
    return agg


def build_placebo_features(months: list[str], k: int = CLUSTER_K, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(months), k))
    return pd.DataFrame(data, index=months, columns=PLACEBO_COLS)


# ===========================================================================
# 모델 zoo (009와 동일)
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
# 핵심: 베이스라인 freeze + 클러스터감성/placebo 잔차보정 증분 평가 (009와 동일 로직)
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

        for variant, cols in (("clsent", CLSENT_COLS), ("placebo", PLACEBO_COLS)):
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
    for variant in ("clsent", "placebo"):
        sub = g[g["variant"] == variant]
        ens = sub[sub["resid_model"] == "ensemble"].set_index("baseline")["ic"]
        best_idx = sub.loc[sub.groupby("baseline")["ic"].idxmax()].set_index("baseline")
        out[f"{variant}_ensemble_ic"] = ens
        out[f"{variant}_ensemble_delta"] = ens - out["base_ic"]
        out[f"{variant}_best_ic"] = best_idx["ic"]
        out[f"{variant}_best_arch"] = best_idx["resid_model"]
        out[f"{variant}_best_delta"] = best_idx["ic"] - out["base_ic"]
    out["clsent_vs_placebo_ensemble"] = out["clsent_ensemble_delta"] - out["placebo_ensemble_delta"]
    out["clsent_vs_placebo_best"] = out["clsent_best_delta"] - out["placebo_best_delta"]
    order = [m for m in BASE_MODEL_ORDER + ["ensemble"] if m in out.index]
    return out.loc[order]


def stage_incremental_val(df: pd.DataFrame, pool_month: np.ndarray, semantic_emb: np.ndarray,
                          sentiment: np.ndarray) -> None:
    header("[5] 베이스라인 freeze + 클러스터감성 잔차보정 증분 평가 (VAL)")
    months_all = sorted(df["month"].unique())
    clsent_feat = build_cluster_sentiment_features(pool_month, semantic_emb, sentiment,
                                                   train_end=SPLITS["train"][1], export_months=months_all)
    placebo_feat = build_placebo_features(months_all)
    dfn = (df.merge(clsent_feat, left_on="month", right_index=True, how="left")
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


def stage_incremental_cv(df: pd.DataFrame, pool_month: np.ndarray, semantic_emb: np.ndarray,
                         sentiment: np.ndarray) -> None:
    header(f"[6] walk-forward CV (val 연도 {CV_YEARS[0]}~{CV_YEARS[-1]}, "
           f"베이스라인 freeze + 클러스터감성 잔차보정)")
    months_all = sorted(df["month"].unique())
    placebo_feat_full = build_placebo_features(months_all)
    all_records = []
    for year in tqdm(CV_YEARS, desc="cv fold"):
        clsent_feat = build_cluster_sentiment_features(pool_month, semantic_emb, sentiment,
                                                       train_end=f"{year - 1}-12", export_months=months_all)
        dfn = (df.merge(clsent_feat, left_on="month", right_index=True, how="left")
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
        cl_ic = mean_ic("ensemble", "ensemble", "clsent")
        pl_ic = mean_ic("ensemble", "ensemble", "placebo")
        print(f"  {year}: base(ensemble)={base_ic:.3f}, "
              f"+clsent(ensemble resid)={cl_ic:.3f} (Δ{cl_ic - base_ic:+.3f}), "
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
    sentiment = stage_sentiment_embed()
    assert len(pool_month) == len(semantic_emb) == len(sentiment), "기사 수가 안 맞음"

    df = pd.read_csv(FEATURES_PATH, dtype={"month": str, "code": str})
    df[TARGET_RANK] = df.groupby("month")[TARGET].rank(pct=True) - 0.5
    stage_incremental_val(df, pool_month, semantic_emb, sentiment)
    stage_incremental_cv(df, pool_month, semantic_emb, sentiment)


if __name__ == "__main__":
    main()
