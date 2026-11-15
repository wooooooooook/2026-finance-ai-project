"""sample_cluster_umap_sbert.py에서 쓰는 공통 플로팅·품질지표 유틸 (bigkinds-crawler/cluster_common.py와 동일)."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def print_cluster_metrics(embeddings: np.ndarray, coords: np.ndarray, labels: np.ndarray, method_name: str = "UMAP") -> None:
    mask = labels != -1
    n_labels = len(set(labels[mask]))

    if n_labels < 2 or mask.sum() < 2:
        print("클러스터 품질 지표 계산 불가 (유효 클러스터가 2개 미만)")
        return

    print(f"클러스터 품질 지표 (노이즈 {(~mask).sum()}개 제외, {mask.sum()}개 대상, {n_labels}개 클러스터)")
    for space_name, space in [("768차원 원본 임베딩", embeddings), (f"{method_name} 2차원 좌표", coords)]:
        sil = silhouette_score(space[mask], labels[mask])
        ch = calinski_harabasz_score(space[mask], labels[mask])
        db = davies_bouldin_score(space[mask], labels[mask])
        print(
            f"  [{space_name}] silhouette={sil:.4f} (1에 가까울수록 뚜렷, 0 근처/음수면 불분명) | "
            f"calinski-harabasz={ch:.1f} (클수록 뚜렷) | "
            f"davies-bouldin={db:.4f} (작을수록 뚜렷)"
        )


def plot_by_category(
    coords: np.ndarray,
    categories: np.ndarray,
    title: str,
    image_path,
    legend_title: str,
    noise_value=None,
    method_name: str = "UMAP",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap("tab20")

    if noise_value is not None and (categories == noise_value).any():
        noise_mask = categories == noise_value
        ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1], s=4, color="lightgray", label="noise")
    else:
        noise_mask = np.zeros(len(categories), dtype=bool)

    unique_values = sorted(set(categories[~noise_mask]))
    for i, value in enumerate(unique_values):
        mask = categories == value
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, color=cmap(i % 20), label=str(value))

    ax.set_title(title)
    ax.set_xlabel(f"{method_name}-1")
    ax.set_ylabel(f"{method_name}-2")
    ax.legend(title=legend_title, markerscale=3, loc="best", ncol=2 if len(unique_values) > 15 else 1)
    fig.tight_layout()
    fig.savefig(image_path, dpi=300)
    plt.close(fig)
