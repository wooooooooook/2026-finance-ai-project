"""
샘플 뉴스를 뽑아 bge-m3-korean으로 임베딩하고 UMAP(기본 파라미터, 2차원) -> HDBSCAN(기본
파라미터)으로 클러스터링한 뒤 연도별/클러스터별 산점도를 이미지로 저장한다. 본 학습
파이프라인(step3_baseline_train_model.py) 전에 이 조합이 잘 도는지 확인하는 용도.

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step0_cluster_visualize.py
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from tqdm import tqdm
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
RAW_NEWS_DIR = PROJECT_ROOT / "raw" / "news"
OUT_DIR = BASE_DIR / "quicktest"
SAMPLE_PATH = OUT_DIR / "samples" / "sampled_pool.csv.gz"

NUM_WORKERS = os.cpu_count() or 4
YEARS = range(2001, 2026)
SAMPLES_PER_YEAR = 300
RANDOM_STATE = 42

MODEL_NAME = "upskyy/bge-m3-korean"
BATCH_SIZE = 32


def sample_year(year: int) -> pd.DataFrame:
    day_files = sorted(RAW_NEWS_DIR.glob(f"bigkinds_{year}-*.csv.gz"))
    frames = [pd.read_csv(f, usecols=["제목", "본문"]) for f in day_files]
    pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["제목", "본문"])

    rng = np.random.default_rng(RANDOM_STATE + year)
    n_sample = min(SAMPLES_PER_YEAR, len(pool))
    idx = rng.choice(len(pool), size=n_sample, replace=False)
    sampled = pool.iloc[idx].copy()
    sampled["year"] = year
    return sampled


def sample_news() -> None:
    if SAMPLE_PATH.exists():
        print(f"샘플 이미 존재함, 건너뜀: {SAMPLE_PATH}")
        return

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"연도 {len(list(YEARS))}개, {NUM_WORKERS}개 프로세스로 풀링+샘플링")

    results: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(sample_year, year): year for year in YEARS}
        with tqdm(total=len(futures), desc="연도별 샘플링") as pbar:
            for future in as_completed(futures):
                year = futures[future]
                df = future.result()
                pbar.set_postfix(year=year, n=len(df))
                pbar.update(1)
                results.append(df)

    combined = pd.concat(results, ignore_index=True).sort_values("year").reset_index(drop=True)

    titles = combined["제목"].fillna("").astype(str)
    bodies = combined["본문"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True)
    combined["text"] = (titles + ". " + bodies).str.strip()

    out = combined[["year", "제목", "text"]].rename(columns={"제목": "title"})
    out.to_csv(SAMPLE_PATH, index=False, compression="gzip")
    print(f"총 {len(out)}개 샘플 저장 -> {SAMPLE_PATH}")


def get_device() -> str:
    forced = os.environ.get("FORCE_DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed_cluster_plot() -> None:
    df = pd.read_csv(SAMPLE_PATH)
    texts = df["text"].fillna("").astype(str).tolist()
    years = df["year"].to_numpy()
    print(f"샘플 수: {len(texts)}")

    device = get_device()
    print(f"model={MODEL_NAME} device={device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True,
    )
    print("임베딩 완료:", embeddings.shape)

    reducer = umap.UMAP(verbose=True)  # 기본 파라미터 (n_neighbors=15, min_dist=0.1, n_components=2)
    coords = reducer.fit_transform(embeddings)
    print("UMAP 완료:", coords.shape)

    print("HDBSCAN 클러스터링 중...")
    clusterer = HDBSCAN()  # 기본 파라미터
    labels = clusterer.fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"클러스터 수: {n_clusters}, 노이즈 비율: {n_noise / len(labels):.1%}")

    np.savez(OUT_DIR / "quicktest_result.npz", years=years, coords=coords, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=years, cmap="viridis", s=8, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="year")
    ax.set_title(f"UMAP projection colored by year (n={len(df)})")
    fig.savefig(OUT_DIR / "quicktest_umap_year.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    is_noise = labels == -1
    ax.scatter(coords[is_noise, 0], coords[is_noise, 1], c="lightgray", s=8, alpha=0.4, label="noise")
    ax.scatter(coords[~is_noise, 0], coords[~is_noise, 1], c=labels[~is_noise], cmap="tab20", s=8, alpha=0.7)
    ax.set_title(f"HDBSCAN clusters (n_clusters={n_clusters}, noise={n_noise / len(labels):.0%})")
    ax.legend()
    fig.savefig(OUT_DIR / "quicktest_umap_cluster.png", dpi=150)
    plt.close(fig)

    print(f"이미지 저장 완료 -> {OUT_DIR}")


def main() -> None:
    sample_news()
    embed_cluster_plot()


if __name__ == "__main__":
    main()
