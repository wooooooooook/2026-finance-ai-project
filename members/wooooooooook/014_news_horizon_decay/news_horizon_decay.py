"""
news_horizon_decay.py: 014 실험 — "뉴스의 증분 예측력은 월별 지평에서 유의하지 않다. 신호가
며칠 안에 가격에 흡수돼 빠르게 소멸하기 때문이다"(연구 메모 1번 결론)를, 일별/월별 지평을
직접 비교해서 검증한다.

raw/news는 기업명 검색이 아니라 경제일간지 카테고리 전체라 그 날/그 달 전 종목에 똑같은 값이
붙는다(008 README 참고). 그래서 종목별 순위(Rank IC)가 아니라 "뉴스 신호 -> 다음 지평 시장
평균 수익률"의 시계열 예측으로 검증한다 — baseline 종목별 팩터 모델이 필요 없고, 이 실험만으로
독립적으로 완결된다.

## 비교 설계: 수익률 단독 vs 수익률+뉴스 (009식 증분가치 틀을 시장 단위로 축소)

"뉴스 단독으로 시장수익률을 맞춘다"가 아니라, **baseline(그 기간 자체의 수익률) 대비 뉴스를
더했을 때 증분 가치가 있는지**를 본다 — 009가 종목 단위로 했던 "baseline freeze + 증분
테스트"를 시장 단위로 가볍게 가져온 것.
  - 모델 A(baseline): 오늘(이번 기간) 수익률 -> 다음 기간 수익률 (단순 자기회귀/모멘텀)
  - 모델 B(+뉴스): 오늘 수익률 + 오늘 뉴스 피처 -> 다음 기간 수익률
  - 두 모델의 차이(B - A)를 일별/월별 지평에서 비교 — 차이가 일별에서만 뚜렷하면 "뉴스는
    빨리 소멸한다"는 가설 지지.
baseline이 `raw/daily_price`/`raw/returns`에서 바로 계산되는 단순 수익률이라 macro나
투자자별 거래(trading flow) 데이터, 정교한 모멘텀(mom_12_2 같은 다구간 스펙) 없이 지금
가진 데이터만으로 끝난다. **macro/trading flow 일별 수집 + 정교한 모멘텀 피처는 015번
실험으로 분리**(008/009 수준의 23개 팩터를 일별로 복제하려면 투자자별 일별 거래데이터·
일별 외국인 보유비율을 새로 수집해야 해서 작업량이 큼 — 014와는 분리).

단계 (이 파일에 전부 구현, 다른 실험 폴더와 import 공유 없음):
  1. [구현됨] 일별 뉴스 전처리 + 10% 무작위 샘플링
  2. [구현됨] 기사 단위 FinBERT [CLS] 임베딩(768차원, 학습 없는 고정 변환) 캐싱
  3. [계획] attention pool(MLP로 기사 중요도 학습 -> 가중합 -> 8차원 투영, 008/009와 동일
     구조). 일별/월별 타깃이 다르므로 이 단계에서 분기됨 — 2번(임베딩)까지는 타깃과 무관하게
     한 번만 하면 됨
  4. [계획] 일별/주별/월별 시장 평균 수익률 시계열 구성 (raw/daily_price, raw/returns) —
     이 값이 뉴스 피처의 예측 대상이면서 동시에 모델 A/B 공통 입력(baseline)으로도 쓰임
  5. [계획] XGBoost 회귀로 모델 A(수익률만) vs 모델 B(수익률+뉴스) 비교, 지평별(일/월) 증분
     가치 비교

## 1단계: 일별 뉴스 전처리 + 샘플링

PREPROCESSING.md 1~4·6번 적용(008 _read_news_day와 동일한 필터):
  제목/본문 빈 행 제거 -> 분석제외 여부 설정 행 제거 -> 제목+본문 완전중복 제거
  -> 비정상 반복 제목("New" 깨짐 등, 파일의 1% 이상 & 10회 이상) 제거
그 다음 008처럼 "일자별 기사 수 비중대로 월 1,000개"가 아니라, **그 날 필터링 통과한 기사의
10%를 그냥 무작위로 샘플링**한다(일별/월별 비교에서 같은 표본을 쓰기 위해 — 날짜별로 독립적으로
10%씩 뽑으므로 일별 집계든 월별 집계든 같은 표본의 부분합이 된다).

텍스트는 008과 동일하게 "제목. 본문[:200자]"(NEWS_BODY_CHARS=200 — 008/009/011/012가 전부
이 길이를 씀. BigKinds 비로그인 수집은 본문이 원래 이 정도만 채워져 있어서, 제목만 쓰면
실제로는 신호가 있는데 텍스트가 짧아 클러스터링/임베딩이 잘 안 돼 "신호 없음"으로 잘못
보일 위험이 있음 — 그래서 200자라도 포함해서 쓴다).

출력: news_cache/daily_samples.csv.gz (날짜, month, text), news_cache/daily_counts.csv
(날짜, month, n_filtered, n_sampled), news_cache/month_counts.csv(월별 합계 — 008의
month_counts.csv와 n_filtered 합계가 같아야 함, 필터 로직이 동일하므로 교차검증 가능).

## 2단계: FinBERT 임베딩

008 `_embed_batch`/`stage_news_embed`와 완전히 동일한 방식(모델 `snunlp/KR-FinBert-SC`,
[CLS] 토큰, max_length=256, batch=32) — 다만 008은 캐시를 `month`로 묶어서 저장했는데,
이 실험은 일별 비교가 목적이라 **`date`(일 단위)로 묶어서** `news_cache/emb_<연도>.npz`에
저장한다(나중에 일별로도 월별로도 그룹핑 가능). 연도 단위 캐시라 중단 후 재실행하면 이미
끝난 연도는 건너뛴다.

이 GPU 없는 서버에서 008 캐시 생성 시각으로 역산한 처리 속도는 초당 약 23~25개 기사.
10% 샘플 전체(약 293만 개) 기준 **약 1.4일** 예상 — tmux로 돌려두고 기다려야 함.

사용법 (프로젝트 루트에서, cd 없이):
    uv run python members/wooooooooook/014_news_horizon_decay/news_horizon_decay.py
(루트 .venv로 실행 가능 — pykrx 등 특수 의존성 없음, torch/transformers는 루트
requirements.txt에 이미 포함돼 있어 별도 가상환경 불필요)
"""

import ctypes
import gc
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.stats import spearmanr
from tqdm import tqdm

# ===========================================================================
# 설정
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
RAW = PROJECT_ROOT / "raw"

NEWS_CACHE_DIR = BASE_DIR / "news_cache"
DAILY_SAMPLES_PATH = NEWS_CACHE_DIR / "daily_samples.csv.gz"
DAILY_COUNTS_PATH = NEWS_CACHE_DIR / "daily_counts.csv"
MONTH_COUNTS_PATH = NEWS_CACHE_DIR / "month_counts.csv"

NEWS_SAMPLE_FRAC = 0.1   # 그 날 필터링 통과한 기사 중 무작위로 남길 비율
NEWS_BODY_CHARS = 200    # 008/009/011/012와 동일
SEED = 42

N_CORES = os.cpu_count() or 1   # 008과 동일하게 전체 코어 사용(1단계 프로세스풀, 2단계 torch 스레드 공용)

# 008/009/011/012와 동일한 FinBERT 임베딩 설정
EMBED_MODEL = "snunlp/KR-FinBert-SC"
EMBED_MAX_LEN = 256
EMBED_BATCH = 32


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ===========================================================================
# 1. 일별 뉴스 전처리 + 10% 무작위 샘플링
#    필터는 008 _read_news_day와 동일(PREPROCESSING.md 1~4·6번):
#      제목/본문 빈 행 제거 -> 분석제외 여부 행 제거 -> 제목+본문 완전중복 제거
#      -> 비정상 반복 제목(파일의 1% 이상 & 10회 이상) 제거
# ===========================================================================

def _read_and_sample_day(path_str: str):
    path = Path(path_str)
    date = path.name.removesuffix(".csv.gz").replace("bigkinds_", "")  # YYYY-MM-DD
    # (path.stem은 확장자를 하나만 벗겨서 "bigkinds_2001-01-01.csv"가 남는 버그가 있었음 —
    # 이미 생성된 news_cache/daily_samples.csv.gz·emb_*.npz에는 그 버그가 섞여 있어서
    # 재생성하지 않는 한 그대로이고, _load_all_article_embeddings()에서 방어적으로 자름)
    month = date[:7]

    d = pd.read_csv(path, usecols=["제목", "본문", "분석제외 여부"], dtype=str,
                     keep_default_na=False, encoding="utf-8-sig")
    d = d[(d["제목"].str.strip() != "") & (d["본문"].str.strip() != "")]
    d = d[d["분석제외 여부"].str.strip() == ""]
    d = d.drop_duplicates(["제목", "본문"])
    vc = d["제목"].value_counts()
    bad = vc[(vc >= 10) & (vc >= 0.01 * len(d))].index
    d = d[~d["제목"].isin(bad)]
    n_filtered = len(d)

    seed = zlib.crc32(path.name.encode()) % (2 ** 32)
    sampled = d.sample(frac=NEWS_SAMPLE_FRAC, random_state=seed) if n_filtered > 0 else d
    n_sampled = len(sampled)

    text = (sampled["제목"].str.strip() + ". "
            + sampled["본문"].str.slice(0, NEWS_BODY_CHARS).str.replace(r"\s+", " ", regex=True))

    return date, month, n_filtered, n_sampled, text.tolist()


def stage_news_daily_sample() -> None:
    header("[1] 일별 뉴스 전처리 + 10% 무작위 샘플링")
    if DAILY_SAMPLES_PATH.exists():
        print(f"이미 존재 -> {DAILY_SAMPLES_PATH} (다시 만들려면 파일을 지우고 재실행)")
        return

    files = sorted(str(p) for p in (RAW / "news").glob("bigkinds_*.csv.gz"))
    print(f"일별 파일 {len(files)}개, 샘플링 비율 {NEWS_SAMPLE_FRAC:.0%}")

    rows, counts = [], []
    with ProcessPoolExecutor(max_workers=N_CORES) as pool:
        for date, month, n_filtered, n_sampled, texts in tqdm(
            pool.map(_read_and_sample_day, files, chunksize=8), total=len(files), desc="news files",
        ):
            rows.extend((date, month, t) for t in texts)
            counts.append({"date": date, "month": month, "n_filtered": n_filtered, "n_sampled": n_sampled})

    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    samples = pd.DataFrame(rows, columns=["date", "month", "text"])
    samples.to_csv(DAILY_SAMPLES_PATH, index=False)
    print(f"저장 완료 -> {DAILY_SAMPLES_PATH}, 기사 {len(samples)}개")

    daily_counts = pd.DataFrame(counts)
    daily_counts.to_csv(DAILY_COUNTS_PATH, index=False)
    print(f"저장 완료 -> {DAILY_COUNTS_PATH}, {len(daily_counts)}일")

    month_counts = (daily_counts.groupby("month")[["n_filtered", "n_sampled"]]
                     .sum().reset_index())
    month_counts.to_csv(MONTH_COUNTS_PATH, index=False)
    print(f"저장 완료 -> {MONTH_COUNTS_PATH}, {len(month_counts)}개월")


# ===========================================================================
# 2. 뉴스 임베딩 — 기사 단위 FinBERT [CLS] 임베딩 (008 stage_news_embed과 동일한 방식,
#    month 대신 date로 묶어서 저장한다는 점만 다름)
#    학습 없는 고정 변환이라 일별/월별 타깃과 무관하게 한 번만 하면 됨.
#    연도 단위로 캐시해서 중단돼도 이어서 실행 가능.
# ===========================================================================

def _device() -> torch.device:
    """cuda > mps > cpu 순으로 사용 가능한 가속기를 자동 감지. 이 서버는 CPU뿐이지만, 이
    파일을 GPU/MPS 있는 다른 기기로 그대로 복사해서 돌려도 알아서 가속되게 하기 위함."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _embed_batch(model, tokenizer, texts: list[str], device: torch.device) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=EMBED_MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    return out.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] 토큰, 분류 헤드 통과 전


def _current_rss_gb() -> float:
    """/proc/self/status의 실시간 RSS(KB) 조회 — 리눅스 전용, 없으면 -1."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6  # KB -> GB
    except FileNotFoundError:
        pass
    return -1.0


def _release_memory() -> None:
    """연도마다 쌓이는 CPU 메모리(glibc malloc 단편화 — 연도를 반복할수록 해제된 메모리를
    OS에 안 돌려줘서 RSS가 계속 불어남, 2026-11-20 서버에서 62.8GB까지 차서 프로세스가
    강제 종료됨. 맥(MPS)에서는 안 생겼던 문제라 CPU/glibc 쪽 원인으로 봄)를 강제로 정리."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # glibc가 아닌 환경(예: macOS)에서는 조용히 건너뜀


SERVER_YEARS = [str(y) for y in range(2001, 2010)]  # 서버(CPU) 담당: 2001~2009
                                                     # 2010~2025는 다른 기기에서 처리해 emb_<연도>.npz로 합류시킴


def stage_news_embed() -> None:
    header("[2] 뉴스 임베딩 (FinBERT, 기사 단위)")
    samples = pd.read_csv(DAILY_SAMPLES_PATH, dtype={"date": str, "month": str})
    years = [y for y in sorted(samples["date"].str[:4].unique()) if y in SERVER_YEARS]
    todo = [y for y in years if not (NEWS_CACHE_DIR / f"emb_{y}.npz").exists()]
    print(f"서버 담당 연도 {len(years)}개 중 임베딩 필요 {len(todo)}개")
    if not todo:
        return

    from transformers import AutoModel, AutoTokenizer

    device = _device()
    print(f"device = {device}")
    if device.type == "cpu":
        torch.set_num_threads(N_CORES)
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    # AutoModel: SequenceClassification 체크포인트에서 분류 헤드는 빼고 베이스 인코더만 로드
    # (분류 헤드 weight가 안 맞는다는 경고가 뜨는데 정상)
    model = AutoModel.from_pretrained(EMBED_MODEL).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    for year in todo:
        part = samples[samples["date"].str.startswith(year)]
        texts = part["text"].fillna("").astype(str).tolist()
        t0 = time.time()
        vecs = [_embed_batch(model, tokenizer, texts[i:i + EMBED_BATCH], device)
                for i in tqdm(range(0, len(texts), EMBED_BATCH), desc=f"finbert {year}")]
        emb = np.concatenate(vecs, axis=0).astype(np.float32) if vecs else np.zeros((0, 768), dtype=np.float32)
        NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(NEWS_CACHE_DIR / f"emb_{year}.npz", date=part["date"].to_numpy().astype(str), embeddings=emb)
        dt = time.time() - t0
        print(f"  {year}: 기사 {len(part)}개, {dt:.0f}s ({len(part) / dt if dt > 0 else 0:.0f}건/초)")

        del part, texts, vecs, emb
        _release_memory()
        print(f"    (메모리 정리 완료, 현재 RSS {_current_rss_gb():.1f}GB)")


# ===========================================================================
# 4. 일별/월별 시장 평균 수익률 시계열
#    raw/daily_price(일별, raw 종가 — 분할/배당 미반영) vs raw/returns(월별,
#    adjusted_monthly_return — 분할/배당 반영) 등가중 평균.
#    raw/daily_price는 분할 시 가격이 뚝 끊기므로, 비정상적으로 큰 일별 변동은 분할로 보고
#    보정(역산)하지 않고 그 날 수익률만 결측 처리한다 — raw/returns의 "결측은 채우지 않는다"
#    원칙과 같은 철학.
#    주의: SPLIT_JUMP_THRESHOLD(25%)는 가격제한폭(최근 ±30%, 2015년 이전 ±15%)과 겹쳐서
#    정상적인 상한가/하한가를 분할로 오인할 수 있음 — 1차 근사치이며 추후 검증 필요.
# ===========================================================================

FEATURES_DIR = BASE_DIR / "features"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"
DAILY_PRICE_DIR = RAW / "daily_price"
RETURNS_DIR = RAW / "returns"

EXCLUDED_CODES = {"004730", "217590"}                          # PREPROCESSING.md 7번 — 사실상 결측
HIGH_MISSING_CODES = {"013890", "005560", "000080", "001780"}  # raw/daily_price README — 거래일 5%+ 누락
OHLC_ANOMALY_CODES = {"000950", "010100", "014790", "015110"}  # raw/daily_price README — 고가/저가 역전
DAILY_EXCLUDED_CODES = EXCLUDED_CODES | HIGH_MISSING_CODES | OHLC_ANOMALY_CODES
SPLIT_JUMP_THRESHOLD = 0.25  # 이 이상 하루 변동은 분할/이상치로 보고 그 날 수익률만 결측 처리(보정 안 함)


def _load_code_order() -> list[str]:
    """raw/returns 파일들과 같은 종목코드 집합·순서(1,086개)를 그대로 따른다."""
    header_row = pd.read_csv(RETURNS_DIR / "train_2001_2021__adjusted_monthly_return.csv",
                              nrows=0, dtype=str)
    return [c for c in header_row.columns if c != "month"]


def _daily_returns_for_ticker(path: Path) -> pd.Series:
    df = pd.read_csv(path, usecols=["날짜", "종가"], dtype={"종가": float})
    df = df.sort_values("날짜")
    ret = df["종가"].pct_change()
    ret[ret.abs() > SPLIT_JUMP_THRESHOLD] = np.nan
    ret.index = df["날짜"].to_numpy()
    return ret


def stage_daily_return_panel() -> None:
    """raw/daily_price -> raw/returns와 같은 포맷(행=날짜/월, 열=종목코드 1,086개 동일 순서,
    결측은 채우지 않음)의 일별 수익률 패널. train_2001_2021/test_2022_2025로 동일하게 분리."""
    header("[4a] 일별 수익률 패널 (raw/returns와 같은 종목코드 포맷)")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    train_path = FEATURES_DIR / "train_2001_2021__daily_return.csv.gz"
    test_path = FEATURES_DIR / "test_2022_2025__daily_return.csv.gz"
    if train_path.exists() and test_path.exists():
        print(f"이미 존재 -> {train_path}, {test_path}")
        return

    codes = _load_code_order()
    series = {}
    for code in tqdm(codes, desc="daily returns"):
        if code in DAILY_EXCLUDED_CODES:
            continue
        p = DAILY_PRICE_DIR / f"{code}.csv"
        if not p.exists():
            continue
        series[code] = _daily_returns_for_ticker(p)

    panel = pd.DataFrame(series).reindex(columns=codes)  # 제외/결측 종목 열도 전부 유지, 값은 NaN
    panel.index.name = "date"
    panel = panel.sort_index()

    train = panel[panel.index <= "2021-12-31"]
    test = panel[panel.index >= "2022-01-01"]
    train.to_csv(train_path)
    test.to_csv(test_path)
    print(f"저장 완료 -> {train_path}({len(train)}일), {test_path}({len(test)}일), "
          f"종목코드 열 {len(codes)}개 유지(그중 {len(series)}개만 실제 값 있음)")


def stage_market_returns() -> None:
    """일별 패널(4a)과 raw/returns를 각각 등가중 평균해서 시장 전체 수익률 시계열 생성."""
    header("[4b] 일별/월별 시장 평균 수익률")
    daily_path = FEATURES_DIR / "market_return_daily.csv"
    monthly_path = FEATURES_DIR / "market_return_monthly.csv"
    if daily_path.exists() and monthly_path.exists():
        print(f"이미 존재 -> {daily_path}, {monthly_path}")
        return

    train = pd.read_csv(FEATURES_DIR / "train_2001_2021__daily_return.csv.gz", index_col="date")
    test = pd.read_csv(FEATURES_DIR / "test_2022_2025__daily_return.csv.gz", index_col="date")
    daily_panel = pd.concat([train, test])
    market_daily = daily_panel.mean(axis=1, skipna=True)
    market_daily.index.name = "date"
    market_daily.name = "market_return"
    market_daily.to_csv(daily_path)
    print(f"저장 완료 -> {daily_path}, {len(market_daily)}일")

    rtrain = pd.read_csv(RETURNS_DIR / "train_2001_2021__adjusted_monthly_return.csv",
                         dtype={"month": str}).set_index("month")
    rtest = pd.read_csv(RETURNS_DIR / "test_2022_2025__adjusted_monthly_return.csv",
                        dtype={"month": str}).set_index("month")
    monthly = pd.concat([rtrain, rtest])
    monthly = monthly.drop(columns=[c for c in EXCLUDED_CODES if c in monthly.columns])
    market_monthly = monthly.mean(axis=1, skipna=True)
    market_monthly.index.name = "month"
    market_monthly.name = "market_return"
    market_monthly.to_csv(monthly_path)
    print(f"저장 완료 -> {monthly_path}, {len(market_monthly)}개월")


# ===========================================================================
# 3. attention pool — 기사 임베딩(768) -> score MLP -> 그룹(거래일/월) 안에서 softmax
#    가중합 -> 투영(N차원) -> head(1차원)으로 "다음 기간 시장수익률" 예측(MSE).
#    학습 끝나면 head는 버리고 N차원 투영값만 뉴스 피처로 씀.
#    세 모델: 일별-8차원, 일별-16차원(표본 많은 일별만 더 큰 차원 테스트), 월별-8차원
#    (월별은 300개월뿐이라 16차원은 과적합 위험만 키워서 뺌).
#
#    일별 그룹핑은 날짜를 그대로 매칭하지 않고 "직전 거래일 다음날 ~ 오늘 거래일" 윈도우로
#    묶는다(주말/휴일 뉴스가 안 빠지게) -- raw/daily_price 기반 거래일 달력(4b에서 만든
#    market_return_daily.csv의 날짜)에 각 기사 날짜를 merge_asof(direction="forward")로
#    매핑하면 바로 이 윈도우 규칙이 됨.
# ===========================================================================

ATTN_HIDDEN = 32
ATTN_LR = 1e-3
ATTN_MAX_EPOCHS = 30
ATTN_PATIENCE = 5  # 연속 이 횟수만큼 val 개선 없으면 조기종료

ATTN_CONFIGS = [
    {"horizon": "daily", "out_dim": 8, "label": "daily_8"},
    {"horizon": "monthly", "out_dim": 8, "label": "monthly_8"},
]
# daily_16(일별 16차원)은 실험해봤으나 단일 분할·롤링 검증 둘 다에서 daily_8보다 일관되게
# 나빴음(결과 섹션 참고) -- 과적합으로 보고 더 안 씀. 과거 결과는
# results/model_comparison.csv, results/walk_forward_cv.csv에 남아있음.

SPLIT_BOUNDS = {  # 008과 동일한 경계. 일별은 날짜로, 월별은 "YYYY-MM"으로 비교
    "train_end": "2017-12-31", "val_end": "2021-12-31",
}


class AttentionPool(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = ATTN_HIDDEN, out_dim: int = 8):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.proj = nn.Linear(in_dim, out_dim)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(emb).squeeze(-1)           # (n_articles,)
        weights = torch.softmax(scores, dim=0)          # 그 그룹 안에서만 정규화
        pooled = (weights.unsqueeze(-1) * emb).sum(dim=0)  # (768,)
        feat = self.proj(pooled)                        # (out_dim,)
        pred = self.head(feat)                           # (1,)
        return feat, pred.squeeze(-1)


def _load_all_article_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """news_cache/emb_<연도>.npz 전부 로드 -> (기사별 날짜 배열, 기사별 768차원 임베딩 배열).
    1단계의 날짜 추출 버그(path.stem이 .gz만 벗겨서 "YYYY-MM-DD.csv"로 저장돼 있음, 이미
    생성된 캐시라 재생성하지 않고 여기서 방어적으로 앞 10자(YYYY-MM-DD)만 취함)."""
    dates, embs = [], []
    for npz_path in sorted(NEWS_CACHE_DIR.glob("emb_*.npz")):
        z = np.load(npz_path)
        dates.append(z["date"].astype(str))
        embs.append(z["embeddings"])
    all_dates = np.concatenate(dates).astype("<U10")
    return all_dates, np.concatenate(embs, axis=0).astype(np.float32)


def _map_dates_to_trading_day(dates: np.ndarray, trading_days: np.ndarray) -> np.ndarray:
    """기사 날짜 -> "그 날짜 이상인 거래일 중 가장 이른 거래일"로 매핑(merge_asof forward).
    = "직전 거래일 다음날 ~ 그 거래일" 윈도우에 속하는 모든 날짜가 그 거래일로 묶임.
    거래일 달력 밖(가장 늦은 거래일보다 미래)인 날짜는 NaN(매칭 실패)으로 남고 이후 버려짐."""
    order = np.argsort(dates, kind="stable")
    sorted_trading_days = np.sort(trading_days)
    # merge_asof는 숫자/datetime형 키만 받아줘서(문자열 StringDtype 불가), 매칭은 datetime으로
    # 하고 결과로 돌려줄 trading_day는 원래 문자열 그대로 둔다.
    left = pd.DataFrame({"date": pd.to_datetime(dates[order])})
    right = pd.DataFrame({"date": pd.to_datetime(sorted_trading_days), "trading_day": sorted_trading_days})
    merged = pd.merge_asof(left, right, on="date", direction="forward")
    result = np.empty(len(dates), dtype=object)
    result[order] = merged["trading_day"].to_numpy()
    return result


def _build_groups(dates: np.ndarray, horizon: str) -> np.ndarray:
    if horizon == "monthly":
        return np.array([d[:7] for d in dates], dtype=object)
    trading_days = pd.read_csv(FEATURES_DIR / "market_return_daily.csv")["date"].to_numpy()
    return _map_dates_to_trading_day(dates, trading_days)


def _load_target_series(horizon: str) -> pd.Series:
    """그룹 키(거래일 날짜 또는 "YYYY-MM") -> 다음 기간 실제 시장수익률(예측 타깃)."""
    path = FEATURES_DIR / ("market_return_daily.csv" if horizon == "daily" else "market_return_monthly.csv")
    key_col = "date" if horizon == "daily" else "month"
    s = pd.read_csv(path, dtype={key_col: str}).set_index(key_col)["market_return"].sort_index()
    return s.shift(-1)  # 그룹 P의 타깃 = P 다음 기간의 실제 수익률


def _split_of(key: str, horizon: str) -> str:
    end_key = SPLIT_BOUNDS["train_end"] if horizon == "daily" else SPLIT_BOUNDS["train_end"][:7]
    val_end = SPLIT_BOUNDS["val_end"] if horizon == "daily" else SPLIT_BOUNDS["val_end"][:7]
    if key <= end_key:
        return "train"
    if key <= val_end:
        return "val"
    return "test"


def _train_one_attention_pool(all_embs: np.ndarray, group_to_idx: dict, targets: pd.Series,
                               horizon: str, out_dim: int, label: str, device: torch.device) -> pd.DataFrame:
    groups = sorted(g for g in group_to_idx if g in targets.index and not pd.isna(targets.get(g)))
    splits = {g: _split_of(g, horizon) for g in groups}
    train_groups = [g for g in groups if splits[g] == "train"]
    val_groups = [g for g in groups if splits[g] == "val"]
    print(f"  [{label}] 그룹 {len(groups)}개 (train {len(train_groups)} / val {len(val_groups)} / "
          f"test {sum(1 for s in splits.values() if s == 'test')})")

    group_tensors = {g: torch.tensor(all_embs[group_to_idx[g]], dtype=torch.float32, device=device)
                      for g in groups}
    target_values = {g: float(targets[g]) for g in groups}

    model = AttentionPool(768, ATTN_HIDDEN, out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=ATTN_LR)

    best_val, best_state, patience = float("inf"), None, 0
    rng = np.random.default_rng(SEED)
    for epoch in range(ATTN_MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(train_groups))
        for i in order:
            g = train_groups[i]
            opt.zero_grad()
            _, pred = model(group_tensors[g])
            loss = (pred - target_values[g]) ** 2
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_mse = np.mean([
                float((model(group_tensors[g])[1] - target_values[g]) ** 2) for g in val_groups
            ]) if val_groups else 0.0
        improved = val_mse < best_val
        if improved:
            best_val, patience = val_mse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        print(f"    epoch {epoch + 1}: val_mse={val_mse:.6f}{' *' if improved else ''}")
        if patience >= ATTN_PATIENCE:
            print(f"    조기종료(epoch {epoch + 1})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        rows = [{"key": g, "split": splits[g],
                 **{f"news_{label}_{i}": v for i, v in enumerate(model(group_tensors[g])[0].cpu().numpy())}}
                for g in groups]
    return pd.DataFrame(rows)


def stage_attention_pool() -> None:
    header("[3] attention pool 학습 (일별-8/16차원, 월별-8차원)")
    device = _device()
    print(f"device = {device}")

    out_paths = {c["label"]: FEATURES_DIR / f"news_feat_{c['label']}.csv" for c in ATTN_CONFIGS}
    if all(p.exists() for p in out_paths.values()):
        print("이미 전부 존재 -> 건너뜀")
        return

    dates, embs = _load_all_article_embeddings()
    print(f"기사 임베딩 {len(dates)}개 로드 완료")

    for config in ATTN_CONFIGS:
        label, horizon, out_dim = config["label"], config["horizon"], config["out_dim"]
        out_path = out_paths[label]
        if out_path.exists():
            print(f"  [{label}] 이미 존재 -> 건너뜀")
            continue

        groups = _build_groups(dates, horizon)
        valid = groups != None  # noqa: E711 (merge_asof 매칭 실패는 None)
        group_to_idx: dict = {}
        for idx, g in zip(np.where(valid)[0], groups[valid]):
            group_to_idx.setdefault(g, []).append(idx)
        group_to_idx = {g: np.array(v) for g, v in group_to_idx.items()}

        targets = _load_target_series(horizon)
        result = _train_one_attention_pool(embs, group_to_idx, targets, horizon, out_dim, label, device)
        result.to_csv(out_path, index=False)
        print(f"  [{label}] 저장 완료 -> {out_path}, {len(result)}개 기간")

        del group_to_idx
        _release_memory()


# ===========================================================================
# 5. 모델 A(수익률만) vs B(수익률+뉴스) vs B_placebo(수익률+랜덤노이즈) 비교
#    "최종" 모델은 A 2개(일별/월별) + B(실제 뉴스) 3개(daily_8/daily_16/monthly_8) = 5개.
#    B_placebo 3개는 보고용이 아니라 "뉴스라서 좋아진 것"인지 확인하는 대조군(009 방식).
#    A는 horizon당 한 번만 학습해서 같은 horizon의 모든 차원 비교에 재사용(표본이 같으므로).
# ===========================================================================

N_TREES, MAX_DEPTH, XGB_LR = 500, 4, 0.05

HORIZON_LABELS = {"daily": ["daily_8"], "monthly": ["monthly_8"]}  # daily_16 제외(위 설명 참고)
N_PLACEBO_REPEATS = 20  # placebo 시드를 여러 개로 반복 -- 단일 시드 결과가 우연인지 확인용


def _placebo_spearman_distribution(df: pd.DataFrame, news_cols: list[str], n_repeats: int = N_PLACEBO_REPEATS) -> np.ndarray:
    """같은 train/test를 유지한 채 placebo 노이즈 시드만 n_repeats번 바꿔 각각 XGBoost로
    평가 -> test_spearman 분포(배열)를 돌려준다."""
    noise_cols = [f"noise_{i}" for i in range(len(news_cols))]
    corrs = []
    for rep in range(n_repeats):
        rep_rng = np.random.default_rng(SEED + rep)
        df_noise = df.copy()
        df_noise[noise_cols] = rep_rng.standard_normal((len(df_noise), len(news_cols)))
        train_n, test_n = df_noise[df_noise["split"] == "train"], df_noise[df_noise["split"] == "test"]
        corrs.append(_fit_eval_xgb(train_n[["lag1_return"] + noise_cols], train_n["target"],
                                    test_n[["lag1_return"] + noise_cols], test_n["target"]))
    return np.array(corrs)


def _fit_eval_xgb(X_train, y_train, X_test, y_test) -> float:
    model = xgb.XGBRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, learning_rate=XGB_LR,
                              objective="reg:squarederror", n_jobs=N_CORES)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    corr, _ = spearmanr(pred, y_test)
    return float(corr)


def stage_model_comparison() -> None:
    header("[5] 모델 A vs B(+뉴스) vs B_placebo 비교")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "model_comparison.csv"
    if out_path.exists():
        print(f"이미 존재 -> {out_path}")
        return

    rows = []
    for horizon, labels in HORIZON_LABELS.items():
        key_col = "date" if horizon == "daily" else "month"
        ret = pd.read_csv(FEATURES_DIR / f"market_return_{horizon}.csv",
                           dtype={key_col: str}).set_index(key_col)["market_return"].sort_index()
        lag1 = ret.rename("lag1_return")
        target = ret.shift(-1).rename("target")

        baseline_fit = False
        for label in labels:
            feat = pd.read_csv(FEATURES_DIR / f"news_feat_{label}.csv", dtype={"key": str}).set_index("key")
            news_cols = [c for c in feat.columns if c.startswith("news_")]

            df = pd.concat([lag1, target], axis=1).join(feat[["split"] + news_cols], how="inner").dropna()
            train, test = df[df["split"] == "train"], df[df["split"] == "test"]
            print(f"  [{horizon}/{label}] train {len(train)} / test {len(test)}")

            if not baseline_fit:
                corr_a = _fit_eval_xgb(train[["lag1_return"]], train["target"],
                                       test[["lag1_return"]], test["target"])
                rows.append({"horizon": horizon, "label": "-", "model": "A_baseline",
                             "test_spearman": corr_a, "test_spearman_std": 0.0, "n_repeats": 1, "final": True})
                baseline_fit = True

            corr_b = _fit_eval_xgb(train[["lag1_return"] + news_cols], train["target"],
                                   test[["lag1_return"] + news_cols], test["target"])
            rows.append({"horizon": horizon, "label": label, "model": "B_news",
                         "test_spearman": corr_b, "test_spearman_std": 0.0, "n_repeats": 1, "final": True})

            placebo_dist = _placebo_spearman_distribution(df, news_cols)
            rows.append({"horizon": horizon, "label": label, "model": "B_placebo",
                         "test_spearman": float(placebo_dist.mean()), "test_spearman_std": float(placebo_dist.std()),
                         "n_repeats": N_PLACEBO_REPEATS, "final": False})
            print(f"    B_placebo({label}) {N_PLACEBO_REPEATS}회 반복: "
                  f"{placebo_dist.mean():.4f} +- {placebo_dist.std():.4f} "
                  f"(min {placebo_dist.min():.4f}, max {placebo_dist.max():.4f})")

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(f"\n저장 완료 -> {out_path}")
    print(result.to_string(index=False))
    print(f"\n('final' 열이 True인 {int(result['final'].sum())}개가 보고용 최종 모델, "
          f"나머지 B_placebo는 뉴스 vs 노이즈 대조군 -- {N_PLACEBO_REPEATS}개 시드 평균±표준편차)")


# ===========================================================================
# 6. 결과 플롯 — results/model_comparison.csv -> images/model_comparison.png
#    일별/월별을 나란히 놓고, 모델(A_baseline/B_news/B_placebo)별 막대로 비교.
#    색은 카테고리 고정 순서(파랑=baseline, 청록=뉴스, 회색=placebo 대조군).
# ===========================================================================

MODEL_COLORS = {"A_baseline": "#2a78d6", "B_news": "#1baf7a", "B_placebo": "#9a9a95"}
MODEL_ORDER = ["A_baseline", "B_news", "B_placebo"]


def stage_plot_results() -> None:
    header("[6] 결과 플롯")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMAGES_DIR / "model_comparison.png"
    result = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    # daily_16처럼 코드에서 제거된 과거 구성은 CSV엔 기록으로 남겨두되 그래프에선 뺌
    active_labels = {lab for labs in HORIZON_LABELS.values() for lab in labs}
    result = result[(result["label"] == "-") | (result["label"].isin(active_labels))]

    horizons = ["daily", "monthly"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    y_min = min(0, result["test_spearman"].min() * 1.15)
    y_max = max(0.05, result["test_spearman"].max() * 1.3)

    for ax, horizon in zip(axes, horizons):
        sub = result[result["horizon"] == horizon].copy()
        # A_baseline은 label이 "-"라 모든 dim 그룹에 공통으로 보여줌
        dim_labels = [l for l in sub["label"].unique() if l != "-"]
        baseline_row = sub[sub["model"] == "A_baseline"]

        x = np.arange(len(dim_labels))
        width = 0.25
        for i, model in enumerate(MODEL_ORDER):
            if model == "A_baseline":
                values = [baseline_row["test_spearman"].iloc[0]] * len(dim_labels) if len(baseline_row) else [0] * len(dim_labels)
                yerr = None
            else:
                rows_by_label = [sub[(sub["label"] == lab) & (sub["model"] == model)] for lab in dim_labels]
                values = [r["test_spearman"].iloc[0] if len(r) else 0.0 for r in rows_by_label]
                yerr = [r["test_spearman_std"].iloc[0] if len(r) and "test_spearman_std" in r else 0.0
                        for r in rows_by_label] if model == "B_placebo" else None
            bars = ax.bar(x + (i - 1) * width, values, width, label=model, color=MODEL_COLORS[model],
                          yerr=yerr, capsize=3, ecolor="#52514e")
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2, v + (0.002 if v >= 0 else -0.008),
                        f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color="#52514e")

        ax.axhline(0, color="#c3c2b7", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(dim_labels)
        ax.set_title(f"{horizon} horizon")
        ax.set_xlabel("news feature dim")
        ax.grid(axis="y", color="#e5e4df", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("test Spearman correlation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("News horizon decay: baseline vs +news vs +placebo", y=1.12, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료 -> {out_path}")


# ===========================================================================
# 7. 시계열 롤링 검증 — attention pool(뉴스 피처)은 재학습하지 않고 그대로 재사용,
#    XGBoost 비교(5단계)만 여러 기간 창(test_year)으로 반복해서 "2022~2025 test 구간
#    하나"가 우연이었는지 확인한다. attention pool까지 폴드별로 재학습하는 008/009식
#    walk-forward CV보다 훨씬 가볍다(그건 폴드당 30~40분, 이건 폴드당 수 초~수십 초).
#    placebo도 폴드마다 여러 시드로 반복.
# ===========================================================================

CV_TEST_YEARS = [2005, 2009, 2013, 2017, 2021, 2025]
N_CV_PLACEBO_REPEATS = 5


def stage_walk_forward_cv() -> None:
    header("[7] 시계열 롤링 검증 (뉴스 피처 재사용, XGBoost만 폴드별 재학습)")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "walk_forward_cv.csv"
    if out_path.exists():
        print(f"이미 존재 -> {out_path}")
        return

    rows = []
    for horizon, labels in HORIZON_LABELS.items():
        key_col = "date" if horizon == "daily" else "month"
        ret = pd.read_csv(FEATURES_DIR / f"market_return_{horizon}.csv",
                           dtype={key_col: str}).set_index(key_col)["market_return"].sort_index()
        lag1 = ret.rename("lag1_return")
        target = ret.shift(-1).rename("target")

        for label in labels:
            feat = pd.read_csv(FEATURES_DIR / f"news_feat_{label}.csv", dtype={"key": str}).set_index("key")
            news_cols = [c for c in feat.columns if c.startswith("news_")]
            noise_cols = [f"noise_{i}" for i in range(len(news_cols))]
            df = pd.concat([lag1, target], axis=1).join(feat[news_cols], how="inner").dropna().sort_index()

            for test_year in CV_TEST_YEARS:
                test_start = f"{test_year}-01-01" if horizon == "daily" else f"{test_year}-01"
                test_end = f"{test_year}-12-31" if horizon == "daily" else f"{test_year}-12"
                train = df[df.index < test_start]
                test = df[(df.index >= test_start) & (df.index <= test_end)]
                if len(train) < 50 or len(test) < 5:
                    print(f"  [{horizon}/{label}/{test_year}] 표본 부족(train {len(train)}, "
                          f"test {len(test)}) -> 건너뜀")
                    continue

                corr_a = _fit_eval_xgb(train[["lag1_return"]], train["target"],
                                       test[["lag1_return"]], test["target"])
                corr_b = _fit_eval_xgb(train[["lag1_return"] + news_cols], train["target"],
                                       test[["lag1_return"] + news_cols], test["target"])

                placebo_dist = []
                for rep in range(N_CV_PLACEBO_REPEATS):
                    rep_rng = np.random.default_rng(SEED + rep + test_year)
                    tr_n, te_n = train.copy(), test.copy()
                    tr_n[noise_cols] = rep_rng.standard_normal((len(tr_n), len(news_cols)))
                    te_n[noise_cols] = rep_rng.standard_normal((len(te_n), len(news_cols)))
                    placebo_dist.append(_fit_eval_xgb(tr_n[["lag1_return"] + noise_cols], tr_n["target"],
                                                      te_n[["lag1_return"] + noise_cols], te_n["target"]))
                placebo_dist = np.array(placebo_dist)

                rows.append({"horizon": horizon, "label": label, "test_year": test_year,
                             "n_train": len(train), "n_test": len(test),
                             "A_baseline": corr_a, "B_news": corr_b,
                             "B_placebo_mean": float(placebo_dist.mean()), "B_placebo_std": float(placebo_dist.std())})
                print(f"  [{horizon}/{label}/{test_year}] n_test={len(test)} A={corr_a:.3f} "
                      f"B_news={corr_b:.3f} B_placebo={placebo_dist.mean():.3f}+-{placebo_dist.std():.3f}")

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(f"\n저장 완료 -> {out_path}")

    summary = result.groupby(["horizon", "label"])[["A_baseline", "B_news", "B_placebo_mean"]].agg(["mean", "std"])
    print("\n=== 폴드 평균(여러 기간 창에 걸친 평균 ± 표준편차) ===")
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "walk_forward_cv_summary.csv")


# ===========================================================================
# 8. 롤링 검증 플롯 — results/walk_forward_cv.csv -> images/walk_forward_cv.png
#    막대 = 6개 기간 평균, 에러바 = 표준편차, 점 = 폴드 하나하나(발표용 "우연이 아니다" 근거).
# ===========================================================================

def stage_plot_walk_forward_cv() -> None:
    header("[8] 롤링 검증 플롯")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMAGES_DIR / "walk_forward_cv.png"
    df = pd.read_csv(RESULTS_DIR / "walk_forward_cv.csv")
    active_labels = {lab for labs in HORIZON_LABELS.values() for lab in labs}
    df = df[df["label"].isin(active_labels)]

    horizons = ["daily", "monthly"]
    models = ["A_baseline", "B_news", "B_placebo_mean"]
    model_display = {"A_baseline": "A_baseline", "B_news": "B_news", "B_placebo_mean": "B_placebo"}
    bar_colors = [MODEL_COLORS["A_baseline"], MODEL_COLORS["B_news"], MODEL_COLORS["B_placebo"]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    all_vals = pd.concat([df[m] for m in models])
    y_min = min(0, all_vals.min() * 1.3)
    y_max = max(0.05, all_vals.max() * 1.3)
    rng_jitter = np.random.default_rng(0)

    for ax, horizon in zip(axes, horizons):
        sub = df[df["horizon"] == horizon]
        x = np.arange(len(models))
        means = [sub[m].mean() for m in models]
        stds = [sub[m].std() for m in models]
        ax.bar(x, means, 0.5, yerr=stds, capsize=4, color=bar_colors, ecolor="#52514e")
        for xi, m in zip(x, models):
            ys = sub[m].to_numpy()
            xs = xi + rng_jitter.uniform(-0.12, 0.12, size=len(ys))
            ax.scatter(xs, ys, color="#1a1a19", s=14, alpha=0.5, zorder=5)

        ax.axhline(0, color="#c3c2b7", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([model_display[m] for m in models])
        ax.set_title(f"{horizon} horizon (n={len(sub)} folds)")
        ax.grid(axis="y", color="#e5e4df", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("test Spearman correlation")
    fig.suptitle("Walk-forward robustness check (6 rolling test years, dots = individual folds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료 -> {out_path}")


if __name__ == "__main__":
    stage_news_daily_sample()
    stage_news_embed()
    stage_daily_return_panel()
    stage_market_returns()
    stage_attention_pool()
    stage_model_comparison()
    stage_plot_results()
    stage_walk_forward_cv()
    stage_plot_walk_forward_cv()
