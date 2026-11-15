"""cluster_umap_monthly.py에서 쓰는 공통 플로팅·품질지표 유틸."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

MAX_LEGEND_CATEGORIES = 30


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
    # 그리는 순서를 섞어서 특정 카테고리(예: 가장 나중에 그려지는 연도)가
    # 항상 다른 점들을 덮어버리는 겹쳐그리기(z-order) 편향을 없앤다.
    order = np.random.default_rng(0).permutation(len(categories))
    coords = coords[order]
    categories = categories[order]

    unique_values = sorted(v for v in set(categories) if v != noise_value)
    cmap = plt.get_cmap("tab20")
    color_map = {v: cmap(i % 20) for i, v in enumerate(unique_values)}
    if noise_value is not None:
        color_map[noise_value] = (0.85, 0.85, 0.85, 1.0)

    colors = [color_map[v] for v in categories]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=4, c=colors)
    ax.set_title(title)
    ax.set_xlabel(f"{method_name}-1")
    ax.set_ylabel(f"{method_name}-2")

    if len(unique_values) <= MAX_LEGEND_CATEGORIES:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[v], label=str(v), markersize=8)
            for v in unique_values
        ]
        if noise_value is not None and (categories == noise_value).any():
            handles.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[noise_value], label="noise", markersize=8)
            )
        ax.legend(handles=handles, title=legend_title, loc="best", ncol=2 if len(handles) > 15 else 1)
    else:
        ax.text(
            0.99, 0.01,
            f"{len(unique_values)}개 카테고리 (범례 생략)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="gray",
        )

    fig.tight_layout()
    fig.savefig(image_path, dpi=300)
    plt.close(fig)
