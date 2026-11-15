"""
sample_and_embed.py로 만든 월별 샘플 임베딩(.npz, 월별 3000개, 제목+본문, KR-SBERT)을
모두 모아 UMAP과 t-SNE 두 가지 방식으로 각각 2차원 축소하고, 각 투영 결과 위에서
HDBSCAN으로 클러스터링한 뒤 연도별/클러스터별 색상으로 구분한 산점도 이미지 총 4장과
클러스터 품질 지표를 출력한다.
"""

import re

import numpy as np
import pandas as pd
import umap
from pathlib import Path
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
from tqdm import tqdm

from cluster_common import plot_by_category, print_cluster_metrics

YEAR_PATTERN = re.compile(r"bigkinds_(\d{4})-")

EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
CSV_PATH = EMBEDDINGS_DIR / "umap_clusters_monthly.csv"

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
TSNE_PERPLEXITY = 30
HDBSCAN_MIN_CLUSTER_SIZE = 15
RANDOM_STATE = 42


def load_all_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    titles, embeddings, years = [], [], []
    npz_paths = sorted(EMBEDDINGS_DIR.glob("*_sampled_embeddings.npz"))
    for npz_path in tqdm(npz_paths, desc="loading embeddings"):
        npz = np.load(npz_path, allow_pickle=True)
        year = int(YEAR_PATTERN.search(npz_path.stem).group(1))
        titles.append(npz["titles"])
        embeddings.append(npz["embeddings"])
        years.append(np.full(npz["titles"].shape[0], year))

    return (
        np.concatenate(titles),
        np.concatenate(embeddings, axis=0),
        np.concatenate(years),
    )


def run_method(method_name: str, coords: np.ndarray, embeddings: np.ndarray, years: np.ndarray) -> np.ndarray:
    print(f"running HDBSCAN on {method_name} coords...")
    clusterer = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE)
    labels = clusterer.fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[{method_name}] found {n_clusters} clusters ({(labels == -1).sum()} noise points)")

    print_cluster_metrics(embeddings, coords, labels, method_name=method_name)

    plot_by_category(
        coords,
        years,
        f"{method_name} projection of monthly-sampled (3000/month) title+body embeddings (KR-SBERT) colored by year "
        f"(n={len(labels)}, clusters={n_clusters})",
        EMBEDDINGS_DIR / f"{method_name.lower()}_clusters_monthly_year.png",
        legend_title="year",
        method_name=method_name,
    )
    plot_by_category(
        coords,
        labels,
        f"{method_name} + HDBSCAN clustering of monthly-sampled (3000/month) title+body embeddings (KR-SBERT) "
        f"(n={len(labels)}, clusters={n_clusters})",
        EMBEDDINGS_DIR / f"{method_name.lower()}_clusters_monthly_cluster.png",
        legend_title="cluster",
        noise_value=-1,
        method_name=method_name,
    )
    print(f"[{method_name}] year-colored image saved to {method_name.lower()}_clusters_monthly_year.png")
    print(f"[{method_name}] cluster-colored image saved to {method_name.lower()}_clusters_monthly_cluster.png")

    return labels


def main() -> None:
    titles, embeddings, years = load_all_embeddings()
    print(f"loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

    print("running UMAP...")
    umap_coords = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=RANDOM_STATE,
        verbose=True,
    ).fit_transform(embeddings)
    print("UMAP done.")

    print("running t-SNE...")
    tsne_coords = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        init="pca",
        random_state=RANDOM_STATE,
        verbose=1,
    ).fit_transform(embeddings)
    print("t-SNE done.")

    umap_labels = run_method("UMAP", umap_coords, embeddings, years)
    tsne_labels = run_method("tSNE", tsne_coords, embeddings, years)

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
    df.to_csv(CSV_PATH, index=False)
    print(f"csv saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
