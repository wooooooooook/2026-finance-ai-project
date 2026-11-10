"""
embed_titles_sbert.py로 생성한 임베딩(.npz, KR-SBERT 문장 의미 임베딩)에서
연도별로 제목 2000개씩 무작위로 뽑아 UMAP과 t-SNE 두 가지 방식으로 각각 2차원 축소하고,
각 투영 결과 위에서 HDBSCAN으로 클러스터링한 뒤 연도별/클러스터별 색상으로 구분한
산점도 이미지 총 4장과 클러스터 품질 지표를 출력한다.
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
CSV_PATH = EMBEDDINGS_DIR / "umap_clusters_sampled.csv"

SAMPLES_PER_YEAR = 2000
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
TSNE_PERPLEXITY = 30
HDBSCAN_MIN_CLUSTER_SIZE = 15
RANDOM_STATE = 42


def load_all_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    titles, embeddings, years = [], [], []
    npz_paths = sorted(EMBEDDINGS_DIR.glob("*_embeddings.npz"))
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


def sample_per_year(titles: np.ndarray, embeddings: np.ndarray, years: np.ndarray, rng: np.random.Generator):
    sampled_titles, sampled_embeddings, sampled_years = [], [], []
    for year in sorted(set(years)):
        idx = np.flatnonzero(years == year)
        n = min(SAMPLES_PER_YEAR, idx.size)
        chosen = rng.choice(idx, size=n, replace=False)
        sampled_titles.append(titles[chosen])
        sampled_embeddings.append(embeddings[chosen])
        sampled_years.append(years[chosen])
        print(f"year {year}: sampled {n} / {idx.size}")

    return (
        np.concatenate(sampled_titles),
        np.concatenate(sampled_embeddings, axis=0),
        np.concatenate(sampled_years),
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
        f"{method_name} projection of {SAMPLES_PER_YEAR}/year sampled title embeddings (KR-SBERT) colored by year "
        f"(n={len(labels)}, clusters={n_clusters})",
        EMBEDDINGS_DIR / f"{method_name.lower()}_clusters_sampled_year.png",
        legend_title="year",
        method_name=method_name,
    )
    plot_by_category(
        coords,
        labels,
        f"{method_name} + HDBSCAN clustering of {SAMPLES_PER_YEAR}/year sampled title embeddings (KR-SBERT) "
        f"(n={len(labels)}, clusters={n_clusters})",
        EMBEDDINGS_DIR / f"{method_name.lower()}_clusters_sampled_cluster.png",
        legend_title="cluster",
        noise_value=-1,
        method_name=method_name,
    )
    print(f"[{method_name}] year-colored image saved to {method_name.lower()}_clusters_sampled_year.png")
    print(f"[{method_name}] cluster-colored image saved to {method_name.lower()}_clusters_sampled_cluster.png")

    return labels


def main() -> None:
    titles, embeddings, years = load_all_embeddings()
    print(f"loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

    rng = np.random.default_rng(RANDOM_STATE)
    titles, embeddings, years = sample_per_year(titles, embeddings, years, rng)
    print(f"sampled {embeddings.shape[0]} embeddings total")

    # shuffle so no single year is consistently drawn on top of another
    order = rng.permutation(embeddings.shape[0])
    titles, embeddings, years = titles[order], embeddings[order], years[order]

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
