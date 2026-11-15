"""
embeddings/{key}_embeddings.npz 전부에 대해 UMAP과 t-SNE 각각으로 2차원 축소한 뒤
HDBSCAN으로 클러스터링하고, 좌표+라벨을 clusters/{key}_clusters.csv에 저장한다.
이미지는 그리지 않는다 (step4_plot.py가 담당) - 계산 비용이 큰 단계와 그리기 단계를
분리해서, 그림만 다시 그려야 할 때 재계산하지 않도록 하기 위함.

MODELS에 있는 모든 모델에 대해 순서대로 처리한다 (임베딩이 없는 모델은 건너뜀).

사용법:
    cd 004_yearly_multi_model
    uv run python step3_cluster.py

    # CPU 코어 일부만 쓰고 싶을 때 (기본값은 -1, 전체 코어 사용)
    N_JOBS=5 uv run python step3_cluster.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE

from cluster_common import print_cluster_metrics

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
CLUSTERS_DIR = BASE_DIR / "clusters"

MODELS = ["sbert", "bge_m3", "kpf_bert"]

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
TSNE_PERPLEXITY = 30
HDBSCAN_MIN_CLUSTER_SIZE = 15
RANDOM_STATE = 42

N_JOBS = int(os.environ.get("N_JOBS", "-1"))  # -1 = 전체 코어 사용


def cluster_one(coords: np.ndarray, embeddings: np.ndarray, method_name: str) -> np.ndarray:
    print(f"  running HDBSCAN on {method_name} coords...")
    labels = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, n_jobs=N_JOBS).fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  [{method_name}] found {n_clusters} clusters ({(labels == -1).sum()} noise points)")
    print_cluster_metrics(embeddings, coords, labels, method_name)
    return labels


def process_model(key: str) -> None:
    embeddings_path = EMBEDDINGS_DIR / f"{key}_embeddings.npz"
    if not embeddings_path.exists():
        print(f"[{key}] 임베딩 없음, 건너뜀: {embeddings_path}")
        return

    output_path = CLUSTERS_DIR / f"{key}_clusters.csv"
    if output_path.exists():
        print(f"[{key}] 이미 존재함, 건너뜀: {output_path}")
        return

    print(f"=== {key} ===")
    npz = np.load(embeddings_path, allow_pickle=True)
    titles, years, embeddings = npz["titles"], npz["years"], npz["embeddings"]
    print(f"로드: {len(titles)}개, dim={embeddings.shape[1]}")

    print("running UMAP...")
    umap_coords = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=True,
    ).fit_transform(embeddings)
    umap_labels = cluster_one(umap_coords, embeddings, "UMAP")

    print("running t-SNE...")
    tsne_coords = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        init="pca",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
    ).fit_transform(embeddings)
    tsne_labels = cluster_one(tsne_coords, embeddings, "tSNE")

    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "title": titles,
            "year": years,
            "umap_x": umap_coords[:, 0],
            "umap_y": umap_coords[:, 1],
            "umap_cluster": umap_labels,
            "tsne_x": tsne_coords[:, 0],
            "tsne_y": tsne_coords[:, 1],
            "tsne_cluster": tsne_labels,
        }
    )
    df.to_csv(output_path, index=False)
    print(f"[{key}] 저장 완료 -> {output_path}")


def main() -> None:
    for key in MODELS:
        process_model(key)


if __name__ == "__main__":
    main()
