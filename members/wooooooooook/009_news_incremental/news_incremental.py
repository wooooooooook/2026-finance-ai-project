"""
news_incremental.py: 009 실험 — 008(news_models)은 베이스 23개 feature와 뉴스 8개를 합쳐서
9개 모델(+앙상블)을 매번 처음부터 다시 학습했다. 그 방식은 val에서 거의 모든 모델이 좋아졌지만
walk-forward CV에서는 대부분 사라지거나 역전됐다 — val 단일 분할에 대한 과적합일 가능성이 높다.

이 실험은 "베이스라인(23개 feature)을 완전히 고정(freeze)한 뒤, 뉴스가 그 위에 orthogonal한
증분 정보를 주는가"만 따로 검증한다.

핵심 설계 포인트 (왜 이렇게 했는지, 비자명한 결정들):

1. 뉴스 feature(news_att1~8)는 기업 뉴스가 아니라 경제일간지 전체라 "그 달 전 종목에 같은 값"이
   붙는다(008과 동일, raw/news README 참고). Rank IC는 그 달 안에서의 상대 순위만 보므로, 월 단위로
   상수인 feature는 그 자체로는 절대 Rank IC에 기여할 수 없다(종목마다 똑같은 값을 더해도 그 달
   순위는 안 바뀜). 뉴스가 기여할 수 있는 유일한 통로는 "다른(종목별로 다른) feature와의 상호작용"
   뿐이다 — 008의 트리 모델들이 news를 joint로 넣었을 때만 약하게라도 permutation importance가
   잡힌 것도 이 때문으로 보인다.
   그래서 baseline을 그냥 freeze하고 "뉴스 8개만"으로 잔차를 학습시키는 건 이론상 항상 0이 나오는
   무의미한 테스트가 된다. 대신 baseline의 "그 달 내 예측 순위"(base_rank, 종목마다 다른 값) 1개를
   같이 넣어서, 2단계 모델이 "base_rank와 뉴스의 상호작용"을 배울 수 있게 했다 — stacking/boosting의
   2단계 모델과 동일한 구조(1단계 파라미터는 전혀 갱신되지 않음).

2. placebo 대조군: 뉴스 8차원 대신 똑같은 구조(월 단위로 전 종목에 같은 값)의 랜덤 노이즈 8차원을
   넣고 똑같이 학습시킨다. "뉴스라서 좋아진 것"과 "어차피 입력을 8개 더 주면(특히 트리 계열은) 생기는
   capacity 증가로 인한 가짜 개선"을 구분하기 위한 것. news_delta - placebo_delta가 실제로 봐야 할
   핵심 숫자다.

3. 2단계(잔차) 모델 아키텍처는 008의 9개 모델 zoo를 거의 그대로 재사용하되, xgb_reg/xgb_rank_target은
   1단계에선 "원 수익률 vs 월별 순위"라는 다른 타겟을 썼지만, 2단계는 어차피 같은 잔차(rank residual)
   하나만 예측하므로 구분할 이유가 없어 "xgb" 하나로 합쳤다(8개 아키텍처 + ensemble).

4. 베이스라인도 9개 아키텍처 + (9개의 랭크 평균인) ensemble까지 총 10개를 그대로 모두 돌린다(요청:
   "오래 걸려도 되니 최대한 많은 모델을 사용"). 베이스라인 10개 x 2단계 9개 x variant(news/placebo)
   2개 = 180개 조합을 VAL 1회 + CV 13-fold 전부에 대해 계산한다. 조합이 많아 실행 시간이 길다
   (CV 1-fold당 베이스라인 9개 + 2단계 9x2x10=180개 모델 피팅).

실행 순서 (1~3은 008과 동일한 로직을 이 폴더 안에 독립적으로 복사):
  1. 종목x월 재무·거래 feature 패널. features/panel.csv.gz가 있으면 재사용
  2. 뉴스 월별 샘플링. news_cache/samples.csv.gz가 있으면 재사용
  3. 뉴스 임베딩(FinBERT, 기사 단위), news_cache/emb_<연도>.npz에 연도별 캐시
  4. attention pool 학습(메인 분할) → 베이스라인 10개 freeze → news/placebo 잔차보정 2단계
     모델 학습 → VAL 평가
  5. walk-forward CV(13-fold): fold마다 attention pool 재학습(고정 epoch) + 베이스라인/2단계
     전부 재학습 → 평가
  6. 요약 테이블(CSV) + "news 개선분 - placebo 개선분" 히트맵 이미지

사용법:
    cd members/wooooooooook/009_news_incremental
    python3 news_incremental.py
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
from tqdm import tqdm

# ===========================================================================
# 설정
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
RAW = PROJECT_ROOT / "raw"

FEATURES_PATH = BASE_DIR / "features" / "panel.csv.gz"
NEWS_CACHE_DIR = BASE_DIR / "news_cache"
NEWS_SAMPLES_PATH = NEWS_CACHE_DIR / "samples.csv.gz"
NEWS_COUNTS_PATH = NEWS_CACHE_DIR / "month_counts.csv"
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

# 뉴스 샘플링 (008과 동일)
NEWS_PER_MONTH = 1000
NEWS_PER_DAY_CAP = 300
NEWS_BODY_CHARS = 200

# 뉴스 임베딩 + attention pooling (008과 동일)
EMBED_MODEL = "snunlp/KR-FinBert-SC"
EMBED_MAX_LEN = 256
EMBED_BATCH = 32
NEWS_DIM = 8
NEWS_COLS = [f"news_att{i}" for i in range(1, NEWS_DIM + 1)]
ATTN_HIDDEN = 32
ATTN_LR = 1e-3
ATTN_MAX_EPOCHS = 30

# placebo: news와 똑같이 "그 달 전 종목에 같은 값"인 랜덤 노이즈 8차원
PLACEBO_COLS = [f"placebo{i}" for i in range(1, NEWS_DIM + 1)]

MONTHLY_COLS = set(MACRO_COLS + NEWS_COLS + PLACEBO_COLS)  # 그 달 전 종목에 같은 값인 feature

TARGET = "target_next"
TARGET_RANK = "target_rank"

N_CORES = os.cpu_count() or 1
SEED = 42
N_TREES, DEPTH, LR = 500, 4, 0.05

SCORE_FULL_IC = 0.06

BASE_MODEL_ORDER = [
    "ridge", "mlp", "random_forest", "extra_trees",
    "xgb_reg", "xgb_rank_target", "xgb_pairwise", "lgbm_reg", "catboost_reg",
]
RESID_MODEL_ORDER = [
    "ridge", "mlp", "random_forest", "extra_trees", "xgb", "xgb_pairwise", "lgbm_reg", "catboost_reg",
]


# ===========================================================================
# 공용 함수 (008과 동일)
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


def summarize(ic: pd.Series) -> dict:
    std = ic.std()
    return {
        "ic_mean": ic.mean(),
        "ic_ir": ic.mean() / std if std else np.nan,
        "hit_rate": (ic > 0).mean(),
        "score": (np.clip(ic / SCORE_FULL_IC, 0, 1) * 5).mean(),
        "n_months": len(ic),
    }


# ===========================================================================
# 1. 재무·거래 feature 패널 (008과 동일 — 이 폴더 안에서 독립적으로 재생성)
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
        print(f"이미 존재 -> {FEATURES_PATH} (feature 정의를 바꿨다면 이 파일을 지우고 실행)")
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
# 2. 뉴스 샘플링 (008과 동일)
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
    if NEWS_SAMPLES_PATH.exists():
        print(f"이미 존재 -> {NEWS_SAMPLES_PATH}")
        return

    files = sorted(str(p) for p in (RAW / "news").glob("bigkinds_*.csv.gz"))
    print(f"일별 파일 {len(files)}개")
    rng = np.random.default_rng(SEED)
    rows, counts = [], []
    cur_month, days = None, []

    def flush():
        if cur_month is None:
            return
        sampled = _sample_month(days, rng)
        rows.extend((cur_month, t) for t in sampled)
        counts.append({"month": cur_month, "n_articles": sum(n for n, _ in days), "n_sampled": len(sampled)})

    with ProcessPoolExecutor(max_workers=N_CORES) as pool:
        for month, n, texts in tqdm(pool.map(_read_news_day, files, chunksize=8),
                                    total=len(files), desc="news files"):
            if month != cur_month:
                flush()
                cur_month, days = month, []
            days.append((n, texts))
        flush()

    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["month", "text"]).to_csv(NEWS_SAMPLES_PATH, index=False)
    c = pd.DataFrame(counts)
    c.to_csv(NEWS_COUNTS_PATH, index=False)
    print(f"저장 완료 -> {NEWS_SAMPLES_PATH}, 기사 {len(rows)}개, {len(c)}개월")


# ===========================================================================
# 3. 뉴스 임베딩 (008과 동일)
# ===========================================================================

def _embed_batch(model, tokenizer, texts: list[str]) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=EMBED_MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    return out.last_hidden_state[:, 0, :].numpy()


def stage_news_embed() -> dict[str, torch.Tensor]:
    header("[3] 뉴스 임베딩 (FinBERT, 기사 단위)")
    samples = pd.read_csv(NEWS_SAMPLES_PATH, dtype={"month": str})
    years = sorted(samples["month"].str[:4].unique())
    todo = [y for y in years if not (NEWS_CACHE_DIR / f"emb_{y}.npz").exists()]
    print(f"연도 {len(years)}개 중 임베딩 필요 {len(todo)}개")

    if todo:
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(N_CORES)
        tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        model = AutoModel.from_pretrained(EMBED_MODEL).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        for year in todo:
            part = samples[samples["month"].str.startswith(year)]
            texts = part["text"].fillna("").astype(str).tolist()
            t0 = time.time()
            vecs = [_embed_batch(model, tokenizer, texts[i:i + EMBED_BATCH])
                    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc=f"finbert {year}")]
            emb = np.concatenate(vecs, axis=0).astype(np.float32)
            NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(NEWS_CACHE_DIR / f"emb_{year}.npz", month=part["month"].to_numpy().astype(str), embeddings=emb)
            print(f"  {year}: 기사 {len(part)}개, {time.time() - t0:.0f}s "
                  f"({len(part) / (time.time() - t0):.0f}건/초)")

    emb_by_month: dict[str, torch.Tensor] = {}
    for y in years:
        z = np.load(NEWS_CACHE_DIR / f"emb_{y}.npz", allow_pickle=True)
        months, emb = z["month"].astype(str), z["embeddings"]
        for m in np.unique(months):
            emb_by_month[m] = torch.tensor(emb[months == m], dtype=torch.float32)
    dim = next(iter(emb_by_month.values())).shape[1]
    print(f"월별 기사 임베딩 로드: {len(emb_by_month)}개월, {dim}차원")
    return emb_by_month


# ===========================================================================
# 4. attention pooling (008과 동일)
# ===========================================================================

class AttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden: int = ATTN_HIDDEN, out_dim: int = NEWS_DIM):
        super().__init__()
        self.score_mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.project = nn.Linear(in_dim, out_dim)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, month_embeddings: torch.Tensor):
        scores = self.score_mlp(month_embeddings).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        pooled = (weights.unsqueeze(-1) * month_embeddings).sum(0)
        projected = self.project(pooled)
        pred = self.head(projected)
        return projected, pred


def _epoch_loss(model, months, emb_by_month, target_by_month, optimizer=None):
    total_loss, n = 0.0, 0
    for month in months:
        if month not in emb_by_month or month not in target_by_month:
            continue
        emb, targets = emb_by_month[month], target_by_month[month]
        _, pred = model(emb)
        loss = ((pred - targets) ** 2).mean()
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n if n else float("nan")


def fit_news_features(emb_by_month: dict, target_by_month: dict, train_months: list[str],
                      monitor_months: list[str], export_months: list[str],
                      epochs: int | None = None, seed: int = SEED):
    assert monitor_months or epochs, "epochs 고정값이 없으면 monitor_months(조기종료 기준)가 필요함"
    torch.manual_seed(seed)
    in_dim = next(iter(emb_by_month.values())).shape[1]
    model = AttentionPool(in_dim=in_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=ATTN_LR)

    best_state, best_loss, best_epoch = None, float("inf"), epochs or ATTN_MAX_EPOCHS
    for epoch in range(epochs or ATTN_MAX_EPOCHS):
        model.train()
        _epoch_loss(model, train_months, emb_by_month, target_by_month, optimizer)
        if monitor_months:
            model.eval()
            with torch.no_grad():
                val_loss = _epoch_loss(model, monitor_months, emb_by_month, target_by_month)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    rows, idx = [], []
    with torch.no_grad():
        for m in export_months:
            if m in emb_by_month:
                projected, _ = model(emb_by_month[m])
                rows.append(projected.numpy())
                idx.append(m)
    feat = pd.DataFrame(rows, index=idx, columns=NEWS_COLS)
    return feat, best_epoch, (best_loss if monitor_months else float("nan"))


def build_news_features(df: pd.DataFrame, emb_by_month: dict, train_end: str,
                        monitor_months: list[str] | None = None, epochs: int | None = None):
    target_by_month = {
        m: torch.tensor(g[TARGET].to_numpy(), dtype=torch.float32)
        for m, g in df.groupby("month") if m in emb_by_month
    }
    train_months = [m for m in emb_by_month if SPLITS["train"][0] <= m <= train_end]
    export_months = sorted(emb_by_month)
    return fit_news_features(emb_by_month, target_by_month, train_months,
                             monitor_months or [], export_months, epochs=epochs)


def build_placebo_features(months: list[str], seed: int = SEED) -> pd.DataFrame:
    """news_att1~8과 똑같은 구조(월마다 전 종목 공통값)의 랜덤 노이즈 — placebo 대조군."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(months), len(PLACEBO_COLS)))
    return pd.DataFrame(data, index=months, columns=PLACEBO_COLS)


# ===========================================================================
# 모델 zoo
#   베이스라인: 008과 동일한 9개 아키텍처 (BASE_FEATURES -> TARGET/TARGET_RANK)
#   2단계(잔차): 같은 아키텍처를 ["base_rank"] + news/placebo 8개 -> "resid" 예측에 재사용
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
# 5. 핵심: 베이스라인 freeze + news/placebo 잔차보정 증분 평가
# ===========================================================================

def evaluate_incremental(train_df: pd.DataFrame, eval_df: pd.DataFrame, verbose: bool = True) -> list[dict]:
    """train_df/eval_df는 BASE_FEATURES + NEWS_COLS + PLACEBO_COLS + TARGET + TARGET_RANK를 포함해야 함.
    반환: [{"baseline":.., "resid_model":.., "variant":.., "month":.., "ic":..}, ...] 롱포맷."""
    base_fits = fit_models(make_base_models(BASE_FEATURES), train_df, verbose=verbose, label="baseline")
    base_pred_train = {m: mod.predict(train_df) for m, mod in base_fits.items()}
    base_pred_eval = {m: mod.predict(eval_df) for m, mod in base_fits.items()}
    base_pred_train["ensemble"] = ensemble_pred(base_pred_train, train_df["month"])
    base_pred_eval["ensemble"] = ensemble_pred(base_pred_eval, eval_df["month"])

    target_rank_train = rank_pct(train_df[TARGET].to_numpy(), train_df["month"])
    records = []

    baseline_names = BASE_MODEL_ORDER + ["ensemble"]
    resid_iter = tqdm(baseline_names, desc="baseline", leave=False) if verbose else baseline_names
    for bm in resid_iter:
        base_rank_tr = rank_pct(base_pred_train[bm], train_df["month"])
        base_rank_ev = rank_pct(base_pred_eval[bm], eval_df["month"])

        ic = monthly_ic(eval_df["month"], base_rank_ev, eval_df[TARGET])
        records += [{"baseline": bm, "resid_model": "(base only)", "variant": "base", "month": mo, "ic": v}
                    for mo, v in ic.items()]

        resid_tr = target_rank_train - base_rank_tr
        tr_aux = pd.DataFrame({"month": train_df["month"].to_numpy(), "base_rank": base_rank_tr, "resid": resid_tr})
        ev_aux = pd.DataFrame({"month": eval_df["month"].to_numpy(), "base_rank": base_rank_ev})

        for variant, cols in (("news", NEWS_COLS), ("placebo", PLACEBO_COLS)):
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
    for variant in ("news", "placebo"):
        sub = g[g["variant"] == variant]
        ens = sub[sub["resid_model"] == "ensemble"].set_index("baseline")["ic"]
        best_idx = sub.loc[sub.groupby("baseline")["ic"].idxmax()].set_index("baseline")
        out[f"{variant}_ensemble_ic"] = ens
        out[f"{variant}_ensemble_delta"] = ens - out["base_ic"]
        out[f"{variant}_best_ic"] = best_idx["ic"]
        out[f"{variant}_best_arch"] = best_idx["resid_model"]
        out[f"{variant}_best_delta"] = best_idx["ic"] - out["base_ic"]
    out["news_vs_placebo_ensemble"] = out["news_ensemble_delta"] - out["placebo_ensemble_delta"]
    out["news_vs_placebo_best"] = out["news_best_delta"] - out["placebo_best_delta"]
    order = [m for m in BASE_MODEL_ORDER + ["ensemble"] if m in out.index]
    return out.loc[order]


def stage_incremental_val(df: pd.DataFrame, emb_by_month: dict) -> int:
    header("[4] attention pool 학습 + 베이스라인 freeze + 잔차보정 증분 평가 (VAL)")
    monitor_months = [m for m in emb_by_month if SPLITS["val"][0] <= m <= SPLITS["val"][1]]
    news_feat, best_epoch, best_loss = build_news_features(
        df, emb_by_month, train_end=SPLITS["train"][1], monitor_months=monitor_months,
    )
    print(f"attention pool: best_epoch={best_epoch}/{ATTN_MAX_EPOCHS}, val_loss={best_loss:.6f}")

    placebo_feat = build_placebo_features(sorted(df["month"].unique()))
    dfn = (df.merge(news_feat, left_on="month", right_index=True, how="left")
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


def stage_incremental_cv(df: pd.DataFrame, emb_by_month: dict, news_epochs: int) -> None:
    header(f"[5] walk-forward CV (val 연도 {CV_YEARS[0]}~{CV_YEARS[-1]}, "
           f"베이스라인 freeze + 잔차보정, attention pool epoch={news_epochs} 고정)")
    placebo_feat_full = build_placebo_features(sorted(df["month"].unique()))
    all_records = []
    for year in tqdm(CV_YEARS, desc="cv fold"):
        news_feat, _, _ = build_news_features(
            df, emb_by_month, train_end=f"{year - 1}-12", epochs=news_epochs,
        )
        dfn = (df.merge(news_feat, left_on="month", right_index=True, how="left")
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
        news_ic = mean_ic("ensemble", "ensemble", "news")
        placebo_ic = mean_ic("ensemble", "ensemble", "placebo")
        print(f"  {year}: base(ensemble)={base_ic:.3f}, "
              f"+news(ensemble resid)={news_ic:.3f} (Δ{news_ic - base_ic:+.3f}), "
              f"+placebo={placebo_ic:.3f} (Δ{placebo_ic - base_ic:+.3f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_df = pd.DataFrame(all_records)
    cv_df.to_csv(RESULTS_DIR / "cv_incremental_ic.csv", index=False)

    summary = summarize_incremental(cv_df)
    summary.to_csv(RESULTS_DIR / "cv_incremental_summary.csv")
    print("\n" + summary.to_string(float_format=lambda x: f"{x:+.4f}" if isinstance(x, float) else str(x)))


# ===========================================================================
# 6. 리포트 — "news 개선분 - placebo 개선분" 히트맵
# ===========================================================================

def _heatmap(data: pd.DataFrame, path: Path, title: str, fmt: str) -> None:
    vals = data.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(0.75 * data.shape[1] + 3, 0.5 * data.shape[0] + 1.8))
    lim = np.nanmax(np.abs(vals)) or 1
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
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


def report_heatmap(ic_df: pd.DataFrame, path: Path, title: str) -> None:
    g = ic_df.groupby(["baseline", "resid_model", "variant"])["ic"].mean().reset_index()
    base = g[g["variant"] == "base"].set_index("baseline")["ic"]
    piv = {}
    for variant in ("news", "placebo"):
        sub = g[g["variant"] == variant]
        t = sub.pivot(index="baseline", columns="resid_model", values="ic")
        piv[variant] = t.sub(base, axis=0)
    delta_vs_placebo = piv["news"] - piv["placebo"]
    order_rows = [m for m in BASE_MODEL_ORDER + ["ensemble"] if m in delta_vs_placebo.index]
    order_cols = [m for m in RESID_MODEL_ORDER + ["ensemble"] if m in delta_vs_placebo.columns]
    _heatmap(delta_vs_placebo.loc[order_rows, order_cols], path, title, "+.3f")


def stage_report() -> None:
    header("[6] 비교 이미지 생성")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    val_ic = pd.read_csv(RESULTS_DIR / "val_incremental_ic.csv")
    report_heatmap(val_ic, IMAGES_DIR / "val_delta_vs_placebo.png",
                    "VAL: (+news resid IC - base) minus (+placebo resid IC - base)\n"
                    "row=frozen baseline, col=2단계 잔차보정 모델")

    cv_ic = pd.read_csv(RESULTS_DIR / "cv_incremental_ic.csv", dtype={"month": str})
    report_heatmap(cv_ic, IMAGES_DIR / "cv_delta_vs_placebo.png",
                    "CV 2009-2021: (+news resid IC - base) minus (+placebo resid IC - base)\n"
                    "row=frozen baseline, col=2단계 잔차보정 모델")
    print(f"이미지 저장 완료 -> {IMAGES_DIR}")


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    stage_features()
    stage_news_sample()
    emb_by_month = stage_news_embed()
    df = pd.read_csv(FEATURES_PATH, dtype={"month": str, "code": str})
    df[TARGET_RANK] = df.groupby("month")[TARGET].rank(pct=True) - 0.5
    best_epoch = stage_incremental_val(df, emb_by_month)
    stage_incremental_cv(df, emb_by_month, best_epoch)
    stage_report()


if __name__ == "__main__":
    main()
