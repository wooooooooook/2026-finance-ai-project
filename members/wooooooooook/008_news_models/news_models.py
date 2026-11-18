"""
news_models.py: 008 실험 — 007의 모델 9개(+앙상블) 각각에 경제일간지 뉴스 feature를 추가했을 때
성능이 얼마나 바뀌는지 비교하고, 뉴스 추가 모델에서 각 팩터의 기여도를 측정한다.

뉴스 feature는 005 step7(FinBERT attention pooling, 005의 뉴스 실험 중 최고 성능)과 같은 방식:
기사 단위 FinBERT 임베딩(768) -> score MLP(768->32->1) -> 그 달 안에서 softmax -> 가중합
-> 투영(768->8) 이 8차원이 최종 feature. 학습 신호는 "그 달 투영 벡터로 다음 달 평균 수익률을
예측"(MSE, 005 step7과 동일) — PCA 같은 비지도 압축 대신, 기사별 가중치 자체를 지도학습한다.
005처럼 기업 뉴스가 아니라 경제일간지 전체를 쓰므로, 이 feature는 매크로처럼 그 달 전 종목에
같은 값이 붙는다(raw/news README 참고 — 기업명 검색이 아니라 카테고리 전체 수집).

attention pool은 분할이 바뀔 때마다(메인 1회 + CV 13-fold) 그 시점까지의 train만으로 다시
학습한다. CV fold는 그 fold의 test 연도를 들여다보지 않도록 조기종료를 쓰지 않고, 메인 분할에서
정한 epoch 수를 고정으로 재사용한다.

실행 순서:
  1. 종목×월 재무·거래 feature 패널 (007과 동일). features/panel.csv.gz가 있으면 재사용
  2. 뉴스 샘플링: raw/news 일별 파일 전처리 → 월별 기사 샘플. news_cache/samples.csv.gz가 있으면 재사용
  3. 뉴스 임베딩: 기사 단위 FinBERT 임베딩, 연도별로 news_cache/emb_<연도>.npz에 캐시
  4. attention pool 학습(메인 분할, VAL로 조기종료) → train 학습 → val 평가(base/news) → news 버전 permutation importance
  5. walk-forward CV: fold마다 attention pool 재학습(고정 epoch) → 모델 학습·평가
  6. 비교 테이블·히트맵 이미지

사용법:
    cd members/wooooooooook/008_news_models
    python3 news_models.py
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

# 뉴스 샘플링 (005 step6과 동일한 방식: 일자별 기사 수 비중대로 월 1,000개)
NEWS_PER_MONTH = 1000
NEWS_PER_DAY_CAP = 300    # 일별 파일에서 먼저 뽑아두는 최대 기사 수 (메모리 제한용)
NEWS_BODY_CHARS = 200

# 뉴스 임베딩 + attention pooling (005 step7과 동일 구조)
EMBED_MODEL = "snunlp/KR-FinBert-SC"
EMBED_MAX_LEN = 256
EMBED_BATCH = 32
NEWS_DIM = 8
NEWS_COLS = [f"news_att{i}" for i in range(1, NEWS_DIM + 1)]
ATTN_HIDDEN = 32
ATTN_LR = 1e-3
ATTN_MAX_EPOCHS = 30

NEWS_GROUPS = {**FEATURE_GROUPS, "news": NEWS_COLS}
VARIANTS = {"base": BASE_FEATURES, "news": BASE_FEATURES + NEWS_COLS}
MONTHLY_COLS = set(MACRO_COLS + NEWS_COLS)  # 그 달 전 종목에 같은 값인 feature

TARGET = "target_next"
TARGET_RANK = "target_rank"

N_CORES = os.cpu_count() or 1
SEED = 42
N_PERM_REPEATS = 3
N_TREES, DEPTH, LR = 500, 4, 0.05

# 과제 정량평가: 월별 IC 0 이하 0점, 0.06 이상 5점 (사이 구간은 선형으로 가정)
SCORE_FULL_IC = 0.06


# ===========================================================================
# 공용 함수
# ===========================================================================

def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


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
        "score": (np.clip(ic / SCORE_FULL_IC, 0, 1) * 5).mean(),
        "ls_spread": ls.mean(),
        "n_months": len(ic),
    }


# ===========================================================================
# 1. 재무·거래 feature 패널 (007과 동일)
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
# 2. 뉴스 샘플링 — PREPROCESSING.md 1~4·6번 적용 (005 step6과 동일한 일자가중 샘플링)
#   일별 파일: 제목/본문 빈 행 제거 → 분석제외 여부 설정 행 제거 → 제목+본문 중복 제거
#            → 비정상 반복 제목(파일의 1% 이상 & 10회 이상) 제거 → 최대 NEWS_PER_DAY_CAP개 추출
#   월별: 일별 추출분을 그날 기사 수 비중으로 가중해 NEWS_PER_MONTH개 추출
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

    # 파일명이 날짜순이고 map은 순서를 유지하므로 월이 바뀔 때마다 그 달을 확정
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
    print(f"월별 전처리 후 기사 수: 최소 {c['n_articles'].min()}, 중앙 {int(c['n_articles'].median())}, "
          f"최대 {c['n_articles'].max()} / 샘플 수 최소 {c['n_sampled'].min()}")


# ===========================================================================
# 3. 뉴스 임베딩 — 기사 단위 FinBERT [CLS] 임베딩 (005 step7_build_news_embeddings.py와 동일).
#   월 평균을 미리 내지 않고 기사별로 보관해야 다음 단계의 attention이 기사마다 다른 가중치를 줄 수 있음.
#   연도 단위로 캐시해서 중단돼도 이어서 실행 가능.
# ===========================================================================

def _embed_batch(model, tokenizer, texts: list[str]) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=EMBED_MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    return out.last_hidden_state[:, 0, :].numpy()  # [CLS] 토큰, 분류 헤드 통과 전


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
        # AutoModel: SequenceClassification 체크포인트에서 분류 헤드는 빼고 베이스 인코더만 로드
        # (분류 헤드 weight가 안 맞는다는 경고가 뜨는데 정상)
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
# 4. attention pooling (005 step7_train_attention_pool.py와 동일 구조)
#   기사 임베딩(768) -> score MLP(768->32->1) -> 월 안에서 softmax -> 가중합(768)
#   -> 투영(768->8) [이 8차원이 최종 뉴스 feature] -> (학습 시에만) head(8->1) -> 다음달 평균
#   수익률 예측, MSE loss. FinBERT 자체는 추론 전용(파라미터 고정), 여기서 학습하는 건 이 작은
#   MLP뿐이다. "다음달 평균 수익률"을 예측하므로 매크로와 같은 월 단위 신호를 학습하는 것이지,
#   종목 간 순위를 직접 배우지는 않는다(기업 뉴스가 아니므로 이 한계는 불가피 — raw/news README).
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
        loss = ((pred - targets) ** 2).mean()  # pred(스칼라)를 그 달 전 종목 타겟에 broadcast
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
    """train_months로 학습. monitor_months가 있으면 거기서 가장 낮은 loss가 나온 epoch을
    선택(조기종료, 005 step7과 동일), 없으면 epochs(고정값, CV fold에서 메인 분할의 best_epoch
    재사용 — 그 fold의 test 연도를 들여다보지 않기 위함)만큼 학습. export_months 전체에 대해
    8차원 feature를 뽑아 DataFrame으로 반환한다."""
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


# ===========================================================================
# 모델 (007과 동일, 사용할 feature 목록을 받도록 일반화)
#   ridge / mlp는 종목 feature를 월별 순위로, 월 단위 feature(매크로·뉴스)는 train 평균·표준편차로
#   표준화. 선형 모델에서 월 단위 feature는 그 달 전 종목에 같은 값을 더할 뿐이라 순위(IC)에 영향 없음
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
    def __init__(self, est, features: list[str], target: str = TARGET,
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
    def __init__(self, features: list[str]):
        self.features = features

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
        self.est.fit(d[self.features], label, qid=qid)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.est.predict(df[self.features]), dtype=float)


def make_models(features: list[str]) -> dict:
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
        "random_forest": Model(RandomForestRegressor(**forest_kw), features, impute=True),
        "extra_trees": Model(ExtraTreesRegressor(**forest_kw), features, impute=True),
        "xgb_reg": Model(xgb.XGBRegressor(**xgb_kw), features),
        "xgb_rank_target": Model(xgb.XGBRegressor(**xgb_kw), features, target=TARGET_RANK),
        "xgb_pairwise": XGBPairwise(features),
        "lgbm_reg": Model(lgb.LGBMRegressor(
            n_estimators=N_TREES, max_depth=DEPTH, num_leaves=2 ** DEPTH - 1, learning_rate=LR,
            min_child_samples=100, random_state=SEED, n_jobs=N_CORES, verbose=-1,
        ), features),
        "catboost_reg": Model(CatBoostRegressor(
            iterations=N_TREES, depth=DEPTH, learning_rate=LR,
            random_seed=SEED, thread_count=N_CORES, verbose=0,
        ), features),
    }


MODEL_ORDER = [
    "ridge", "mlp", "random_forest", "extra_trees",
    "xgb_reg", "xgb_rank_target", "xgb_pairwise", "lgbm_reg", "catboost_reg", "ensemble",
]


def fit_all(train_df: pd.DataFrame, features: list[str], verbose: bool = True) -> dict:
    models = {}
    for name, m in make_models(features).items():
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
# 5. 단일 분할 평가 (base / news) + news 버전 permutation importance
#   월 단위 feature(매크로·뉴스)는 "그 달 전 종목에 같은 값" 구조를 유지하도록 월 단위로 섞음.
#   그룹은 그룹 내 feature를 같은 순서로 함께 섞음
# ===========================================================================

def _permute(df: pd.DataFrame, cols: list[str], rng) -> pd.DataFrame:
    out = df.copy()
    if all(c in MONTHLY_COLS for c in cols):
        months = df["month"].unique()
        mapping = dict(zip(months, rng.permutation(months)))
        src = df.groupby("month")[cols].first()
        out[cols] = src.loc[df["month"].map(mapping)].to_numpy()
    else:
        out[cols] = df[cols].to_numpy()[rng.permutation(len(df))]
    return out


def stage_val(df: pd.DataFrame, emb_by_month: dict) -> int:
    header("[4] attention pool 학습 + train 학습 → val 평가 (base / news)")
    monitor_months = [m for m in emb_by_month if SPLITS["val"][0] <= m <= SPLITS["val"][1]]
    news_feat, best_epoch, best_loss = build_news_features(
        df, emb_by_month, train_end=SPLITS["train"][1], monitor_months=monitor_months,
    )
    print(f"attention pool: best_epoch={best_epoch}/{ATTN_MAX_EPOCHS}, val_loss={best_loss:.6f}")

    dfn = df.merge(news_feat, left_on="month", right_index=True, how="left")
    train_df, val_df = select_split(dfn, "train"), select_split(dfn, "val")
    print(f"train={len(train_df)}, val={len(val_df)}")

    rows, ic_rows = [], []
    news_models, news_base_ic = None, {}
    for variant, feats in VARIANTS.items():
        print(f"\n[{variant}] feature {len(feats)}개")
        models = fit_all(train_df, feats)
        preds = predict_all(models, val_df)
        for name in MODEL_ORDER:
            ic = monthly_ic(val_df["month"], preds[name], val_df[TARGET])
            ls = monthly_long_short(val_df["month"], preds[name], val_df[TARGET])
            s = summarize(ic, ls)
            rows.append({"model": name, "variant": variant, **s})
            ic_rows.append(pd.DataFrame({"model": name, "variant": variant,
                                         "month": ic.index, "ic": ic.to_numpy()}))
            if variant == "news":
                news_base_ic[name] = s["ic_mean"]
        if variant == "news":
            news_models = models

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    val_metrics = pd.DataFrame(rows)
    val_metrics.to_csv(RESULTS_DIR / "val_metrics.csv", index=False)
    pd.concat(ic_rows).to_csv(RESULTS_DIR / "val_monthly_ic.csv", index=False)
    wide = val_metrics.pivot(index="model", columns="variant", values=["ic_mean", "score"]).loc[MODEL_ORDER]
    print("\n" + wide.to_string(float_format=lambda x: f"{x:.4f}"))

    header("[4] news 버전 permutation importance (ΔIC)")
    rng = np.random.default_rng(SEED)
    feats = VARIANTS["news"]
    targets = [("feature", f, [f]) for f in feats]
    targets += [("group", g, cols) for g, cols in NEWS_GROUPS.items()]
    records = []
    for kind, name, cols in tqdm(targets, desc="permutation"):
        drops = {m: [] for m in MODEL_ORDER}
        for _ in range(N_PERM_REPEATS):
            perm_df = _permute(val_df, cols, rng)
            for m, p in predict_all(news_models, perm_df).items():
                drops[m].append(news_base_ic[m] - monthly_ic(perm_df["month"], p, perm_df[TARGET]).mean())
        for m, v in drops.items():
            records.append({"kind": kind, "name": name, "model": m, "delta_ic": float(np.mean(v))})
    imp = pd.DataFrame(records)
    imp.to_csv(RESULTS_DIR / "importance.csv", index=False)
    group_tbl = imp[imp["kind"] == "group"].pivot(index="model", columns="name", values="delta_ic")
    print(group_tbl.loc[MODEL_ORDER].to_string(float_format=lambda x: f"{x:+.4f}"))

    return best_epoch


# ===========================================================================
# 6. walk-forward CV — fold마다 attention pool을 그 fold의 train 기간으로 다시 학습.
#   test 연도를 들여다보지 않도록 조기종료 없이 메인 분할에서 정한 epoch 수(news_epochs)를 고정 사용.
# ===========================================================================

def stage_cv(df: pd.DataFrame, emb_by_month: dict, news_epochs: int) -> None:
    header(f"[5] walk-forward CV (val 연도 {CV_YEARS[0]}~{CV_YEARS[-1]}, base / news, "
           f"attention pool epoch={news_epochs} 고정)")
    rows = []
    for year in tqdm(CV_YEARS, desc="cv fold"):
        news_feat, _, _ = build_news_features(
            df, emb_by_month, train_end=f"{year - 1}-12", epochs=news_epochs,
        )
        dfn = df.merge(news_feat, left_on="month", right_index=True, how="left")
        tr = dfn[(dfn["month"] >= SPLITS["train"][0]) & (dfn["month"] <= f"{year - 1}-12")]
        te = dfn[dfn["month"].str.startswith(str(year))]
        line = []
        for variant, feats in VARIANTS.items():
            preds = predict_all(fit_all(tr, feats, verbose=False), te)
            for name in MODEL_ORDER:
                ic = monthly_ic(te["month"], preds[name], te[TARGET])
                ls = monthly_long_short(te["month"], preds[name], te[TARGET])
                for month in ic.index:
                    rows.append({"model": name, "variant": variant, "month": month,
                                 "ic": ic[month], "ls": ls.get(month, np.nan)})
                if name == "ensemble":
                    line.append(f"{variant} ensemble={ic.mean():.3f}")
        print(f"  {year}: " + ", ".join(line))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "cv_monthly.csv", index=False)
    print(f"저장 완료 -> {RESULTS_DIR}/cv_monthly.csv")


# ===========================================================================
# 7. 리포트
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


def _comparison_table(summary: pd.DataFrame, path: Path) -> None:
    spec = [
        ("val_ic_base", "VAL IC\nbase", ".4f"), ("val_ic_news", "VAL IC\n+news", ".4f"),
        ("cv_ic_base", "CV IC\nbase", ".4f"), ("cv_ic_news", "CV IC\n+news", ".4f"),
        ("cv_ic_delta", "CV IC\nΔ", "+.4f"),
        ("cv_ir_base", "CV IC-IR\nbase", ".3f"), ("cv_ir_news", "CV IC-IR\n+news", ".3f"),
        ("cv_score_base", "CV score\nbase", ".2f"), ("cv_score_news", "CV score\n+news", ".2f"),
        ("cv_score_delta", "CV score\nΔ", "+.2f"),
    ]
    spec = [s for s in spec if s[0] in summary.columns]
    cmap = plt.get_cmap("RdYlGn")
    text, colors = [], []
    for model in summary.index:
        text.append([format(summary.loc[model, c], f) for c, _, f in spec])
        row_colors = []
        for c, _, _ in spec:
            v = summary[c]
            if c.endswith("_delta"):
                lim = v.abs().max() or 1
                x = 0.5 + 0.5 * summary.loc[model, c] / lim
            else:
                x = (summary.loc[model, c] - v.min()) / (v.max() - v.min()) if v.max() > v.min() else 0.5
            row_colors.append((*cmap(0.15 + 0.7 * x)[:3], 0.6))
        colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(1.1 * len(spec) + 2.2, 0.45 * len(summary) + 1.4))
    ax.axis("off")
    tbl = ax.table(cellText=text, cellColours=colors, rowLabels=list(summary.index),
                   colLabels=[label for _, label, _ in spec], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)
    ax.set_title("Base vs +news (CV 2009-2021 156 months; score = mean of clip(IC/0.06,0,1)*5)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def stage_report() -> None:
    header("[6] 비교 이미지 생성")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(index=MODEL_ORDER)

    val = pd.read_csv(RESULTS_DIR / "val_metrics.csv")
    for variant in VARIANTS:
        summary[f"val_ic_{variant}"] = val[val["variant"] == variant].set_index("model")["ic_mean"]

    cv = pd.read_csv(RESULTS_DIR / "cv_monthly.csv", dtype={"month": str})
    for (model, variant), g in cv.groupby(["model", "variant"]):
        s = summarize(g["ic"], g["ls"].dropna())
        summary.loc[model, f"cv_ic_{variant}"] = s["ic_mean"]
        summary.loc[model, f"cv_ir_{variant}"] = s["ic_ir"]
        summary.loc[model, f"cv_score_{variant}"] = s["score"]
    summary["cv_ic_delta"] = summary["cv_ic_news"] - summary["cv_ic_base"]
    summary["cv_score_delta"] = summary["cv_score_news"] - summary["cv_score_base"]
    summary = summary.sort_values("cv_score_news", ascending=False)
    summary.to_csv(RESULTS_DIR / "summary.csv")
    _comparison_table(summary, IMAGES_DIR / "model_comparison.png")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    cv["year"] = cv["month"].str[:4]
    yearly = cv.groupby(["model", "variant", "year"])["ic"].mean().unstack("variant")
    delta = (yearly["news"] - yearly["base"]).unstack("year").loc[summary.index]
    _heatmap(delta, IMAGES_DIR / "cv_yearly_ic_delta.png",
             "CV: yearly mean Rank IC change from adding news (+news − base)", "+.3f")

    imp = pd.read_csv(RESULTS_DIR / "importance.csv")
    for kind in ("feature", "group"):
        t = imp[imp["kind"] == kind].pivot(index="model", columns="name", values="delta_ic")
        t = t.loc[summary.index, t.mean().sort_values(ascending=False).index]
        _heatmap(t, IMAGES_DIR / f"importance_{kind}.png",
                 f"+news models, permutation importance on VAL: IC drop when {kind} is shuffled", ".3f")
    print(f"\n이미지 저장 완료 -> {IMAGES_DIR}")


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    stage_features()
    stage_news_sample()
    emb_by_month = stage_news_embed()
    df = pd.read_csv(FEATURES_PATH, dtype={"month": str, "code": str})
    df[TARGET_RANK] = df.groupby("month")[TARGET].rank(pct=True) - 0.5
    best_epoch = stage_val(df, emb_by_month)
    stage_cv(df, emb_by_month, best_epoch)
    stage_report()


if __name__ == "__main__":
    main()
