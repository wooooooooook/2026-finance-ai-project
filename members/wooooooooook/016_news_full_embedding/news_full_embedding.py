"""
news_full_embedding.py: 016 실험 — raw/news 전체(샘플링 없이 100%)를 날짜별로 필터링 +
FinBERT 임베딩해서 캐싱한다. 014(news_horizon_decay)는 비용 문제로 10% 무작위 샘플만
썼는데, 이건 전체를 다 써보려는 별도 시도라 폴더를 분리함(014 코드는 복사해서 독립적으로
고침 — 다른 실험 폴더 import 금지 원칙).

raw/news가 이 스크립트를 돌리는 기기(서버든 노트북이든)에 이미 있다는 전제로, 중간에
압축 텍스트 파일을 따로 안 만들고 raw 파일에서 바로 필터링 + 임베딩까지 한 번에 한다.

## 필터링 (014와 동일, 샘플링만 뺌)

PREPROCESSING.md 1~4·6번: 제목/본문 빈 행 제거 -> 분석제외 여부 설정 행 제거 ->
제목+본문 완전중복 제거 -> 비정상 반복 제목(파일의 1% 이상 & 10회 이상) 제거.
텍스트는 "제목. 본문[:200자]"(NEWS_BODY_CHARS=200, 008/009/011/012/014와 동일).

## 임베딩 및 저장

- 모델: `snunlp/KR-FinBert-SC`, [CLS] 토큰, max_length=256 (014와 동일)
- **날짜 단위로 저장**(`news_cache/emb_{YYYY-MM-DD}.npz`) — 연 단위보다 훨씬 촘촘하게
  재개 가능. 전체 작업이 며칠 걸리는 걸 감안해 중단 시 손실을 최소화하려는 목적
- cuda > mps > cpu 자동 감지(014와 동일 `_device()`). 어느 기기에서 돌려도 됨
- **시작할 때 배치 크기 자동 튜닝**: 기사 수가 가장 많은 날 샘플로 32/64/128/256을 각각
  측정(워밍업 1회 + 3회 반복)해서 가장 빠른 배치를 골라 전체 작업에 사용(`_calibrate_batch_size`)
- 하루 끝날 때마다 메모리 강제 정리(`gc.collect`+`malloc_trim`, MPS면 `torch.mps.empty_cache()`도)
  — 014에서 겪은 메모리 누수(장시간 반복 시 RSS 누적) 재발 방지
- `news_cache/counts.csv`에 날짜별 n_filtered 기록 -> 끝나고 합계가 29,356,582(014에서 이미
  확인한 전체 필터 통과 기사 수)와 일치하는지 교차검증 가능

## 저장 위치

`raw/`가 아니라 이 실험 폴더 `news_cache/`에 저장(가공본이라 raw 폴더 원칙과 안 맞음 —
FinBERT 모델에 종속된 결과물).

사용법 (프로젝트 루트에서, cd 없이):
    uv run python members/wooooooooook/016_news_full_embedding/news_full_embedding.py
"""

import ctypes
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
RAW = PROJECT_ROOT / "raw"

NEWS_CACHE_DIR = BASE_DIR / "news_cache"
COUNTS_PATH = NEWS_CACHE_DIR / "counts.csv"
EXPECTED_TOTAL_FILTERED = 29_356_582  # 014에서 이미 확인한 전체 필터 통과 기사 수(교차검증용)

NEWS_BODY_CHARS = 200
EMBED_MODEL = "snunlp/KR-FinBert-SC"
EMBED_MAX_LEN = 256
CANDIDATE_BATCHES = [32, 64, 128, 256]  # 시작할 때 이 중 가장 빠른 걸 자동으로 골라서 씀
CALIBRATION_REPEATS = 3  # 후보 배치마다 측정 반복 횟수(워밍업 1회는 별도)
REVERSE_ORDER = False  # --reverse 플래그 기본값. True면 최신 날짜부터(2025->2001) 처리
                       # — test 기간(2022~2025)을 먼저 끝내서 3~5단계를 일찍 시작하고 싶을 때 씀

N_CORES = os.cpu_count() or 1


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _current_rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except FileNotFoundError:
        pass
    return -1.0


def _release_memory(device: torch.device) -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # glibc가 아닌 환경(macOS 등)에서는 조용히 건너뜀
    if device.type == "mps":
        torch.mps.empty_cache()


# ===========================================================================
# 필터링 (014 _read_and_sample_day와 동일, 샘플링 라인만 제거)
# ===========================================================================

def _filter_day(path: Path) -> tuple[str, int, list[str]]:
    date = path.name.removesuffix(".csv.gz").replace("bigkinds_", "")  # YYYY-MM-DD
    # (path.stem은 .gz만 벗겨서 "...csv"가 남는 버그가 있었음 -- 014에서 이미 겪음)

    d = pd.read_csv(path, usecols=["제목", "본문", "분석제외 여부"], dtype=str,
                     keep_default_na=False, encoding="utf-8-sig")
    d = d[(d["제목"].str.strip() != "") & (d["본문"].str.strip() != "")]
    d = d[d["분석제외 여부"].str.strip() == ""]
    d = d.drop_duplicates(["제목", "본문"])
    vc = d["제목"].value_counts()
    bad = vc[(vc >= 10) & (vc >= 0.01 * len(d))].index
    d = d[~d["제목"].isin(bad)]

    text = (d["제목"].str.strip() + ". "
            + d["본문"].str.slice(0, NEWS_BODY_CHARS).str.replace(r"\s+", " ", regex=True))
    return date, len(d), text.tolist()


def _embed_batch(model, tokenizer, texts: list[str], device: torch.device) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=EMBED_MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    return out.last_hidden_state[:, 0, :].cpu().numpy()


def _pick_calibration_sample(todo: list[Path], scan_n: int = 10) -> list[str]:
    """배치 크기 테스트용 샘플 텍스트 — todo 앞쪽 몇 개 날짜 중 기사 수가 가장 많은 날을 사용
    (초반 연도는 기사 수가 적어서 큰 배치를 못 테스트할 수 있어 여러 날 중 고름)."""
    best_texts: list[str] = []
    for path in todo[:scan_n]:
        _, _, texts = _filter_day(path)
        if len(texts) > len(best_texts):
            best_texts = texts
    return best_texts


def _calibrate_batch_size(model, tokenizer, device: torch.device, sample_texts: list[str]) -> int:
    header("[0] 배치 크기 자동 튜닝")
    if not sample_texts:
        print(f"  샘플이 비어 있어 기본값 {CANDIDATE_BATCHES[0]} 사용")
        return CANDIDATE_BATCHES[0]

    best_batch, best_rate = CANDIDATE_BATCHES[0], -1.0
    for b in CANDIDATE_BATCHES:
        if b > len(sample_texts):
            print(f"  batch={b}: 샘플 부족({len(sample_texts)}개)으로 건너뜀")
            continue
        chunk = sample_texts[:b]
        _embed_batch(model, tokenizer, chunk, device)  # 워밍업(최초 호출 오버헤드 제외)
        t0 = time.time()
        for _ in range(CALIBRATION_REPEATS):
            _embed_batch(model, tokenizer, chunk, device)
        dt = time.time() - t0
        rate = (b * CALIBRATION_REPEATS) / dt if dt > 0 else 0
        print(f"  batch={b}: {rate:.0f}건/초")
        if rate > best_rate:
            best_batch, best_rate = b, rate

    print(f"  선택된 배치 크기: {best_batch} ({best_rate:.0f}건/초)")
    return best_batch


def stage_full_embedding(reverse: bool = REVERSE_ORDER) -> None:
    header("[1] 전체 뉴스 필터링 + FinBERT 임베딩 (날짜 단위 캐싱)")
    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted((RAW / "news").glob("bigkinds_*.csv.gz"), reverse=reverse)
    todo = [p for p in files
            if not (NEWS_CACHE_DIR / f"emb_{p.name.removesuffix('.csv.gz').replace('bigkinds_', '')}.npz").exists()]
    print(f"전체 {len(files)}일 중 임베딩 필요 {len(todo)}일 "
          f"({'최신 날짜부터' if reverse else '과거 날짜부터'})")
    if not todo:
        return

    device = _device()
    print(f"device = {device}")

    from transformers import AutoModel, AutoTokenizer

    if device.type == "cpu":
        torch.set_num_threads(N_CORES)
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    batch_size = _calibrate_batch_size(model, tokenizer, device, _pick_calibration_sample(todo))

    counts = []
    if COUNTS_PATH.exists():
        counts = pd.read_csv(COUNTS_PATH).to_dict("records")

    for path in tqdm(todo, desc="전체 날짜"):
        date, n_filtered, texts = _filter_day(path)
        t0 = time.time()
        if texts:
            vecs = [_embed_batch(model, tokenizer, texts[i:i + batch_size], device)
                    for i in range(0, len(texts), batch_size)]
            emb = np.concatenate(vecs, axis=0).astype(np.float32)
        else:
            emb = np.zeros((0, 768), dtype=np.float32)
        np.savez(NEWS_CACHE_DIR / f"emb_{date}.npz", embeddings=emb)
        dt = time.time() - t0

        counts.append({"date": date, "n_filtered": n_filtered})
        pd.DataFrame(counts).to_csv(COUNTS_PATH, index=False)

        _release_memory(device)
        rate = n_filtered / dt if dt > 0 else 0
        print(f"  {date}: 기사 {n_filtered}개, {dt:.0f}s ({rate:.0f}건/초), "
              f"RSS {_current_rss_gb():.1f}GB")

    total = sum(c["n_filtered"] for c in counts)
    print(f"\n누적 기사 수 {total}개 (기존 014 전체 필터 통과 수 {EXPECTED_TOTAL_FILTERED}개와 "
          f"{'일치' if total == EXPECTED_TOTAL_FILTERED else '불일치 -- 확인 필요'})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", action="store_true", default=REVERSE_ORDER,
                        help="최신 날짜(2025)부터 거꾸로 처리 (기본: 과거부터)")
    args = parser.parse_args()

    stage_full_embedding(reverse=args.reverse)
