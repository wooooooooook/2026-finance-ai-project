"""step4_plot.py에서 쓰는 플로팅 유틸.

10만 건 단위라 그대로 다 찍으면 겹쳐서 안 보이므로:
- 그리는 순서를 섞어서 특정 카테고리가 항상 위에 덮이는 z-order 편향을 없앰
- 점 자체를 MAX_PLOT_POINTS개로 무작위 서브샘플링해서 그림 (저장되는 CSV는 전체 유지)
- 투명도(alpha)를 줘서 밀집된 부분이 자연스럽게 진하게 보이도록 함
- 고해상도(dpi=600)로 저장
"""

import numpy as np
import matplotlib.pyplot as plt

MAX_PLOT_POINTS = 20000
MAX_LEGEND_CATEGORIES = 30
DPI = 600
POINT_SIZE = 8
ALPHA = 0.4
PLOT_RANDOM_STATE = 0

# 극단적 이상치 때문에 빈 공간이 넓어지는 걸 막기 위해, 축 범위를 퍼센타일 기준으로
# 잘라서 보여준다 (이 범위 밖 점은 그려지긴 하지만 화면에는 안 보임).
AXIS_CLIP_PERCENTILE = 1  # 위아래 각각 1%씩 잘라냄
AXIS_PADDING_RATIO = 0.05  # 잘라낸 범위에 여백 5% 추가


def _clipped_range(values: np.ndarray) -> tuple[float, float]:
    low = np.percentile(values, AXIS_CLIP_PERCENTILE)
    high = np.percentile(values, 100 - AXIS_CLIP_PERCENTILE)
    pad = (high - low) * AXIS_PADDING_RATIO
    return low - pad, high + pad


def plot_by_category(
    coords: np.ndarray,
    categories: np.ndarray,
    title: str,
    image_path,
    legend_title: str,
    noise_value=None,
    method_name: str = "UMAP",
) -> None:
    rng = np.random.default_rng(PLOT_RANDOM_STATE)

    n = len(categories)
    if n > MAX_PLOT_POINTS:
        plot_idx = rng.choice(n, size=MAX_PLOT_POINTS, replace=False)
    else:
        plot_idx = np.arange(n)

    # 서브샘플링 후, 그리는 순서도 섞어서 z-order 편향을 없앤다.
    plot_idx = rng.permutation(plot_idx)
    coords = coords[plot_idx]
    categories = categories[plot_idx]

    unique_values = sorted(v for v in set(categories) if v != noise_value)
    cmap = plt.get_cmap("tab20")
    color_map = {v: cmap(i % 20) for i, v in enumerate(unique_values)}
    if noise_value is not None:
        color_map[noise_value] = (0.85, 0.85, 0.85, 1.0)

    colors = [color_map[v] for v in categories]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=POINT_SIZE, c=colors, alpha=ALPHA, linewidths=0)
    ax.set_xlim(_clipped_range(coords[:, 0]))
    ax.set_ylim(_clipped_range(coords[:, 1]))
    ax.set_title(f"{title} (plotted {len(plot_idx)}/{n})")
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
            f"{len(unique_values)} categories (legend omitted)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="gray",
        )

    fig.tight_layout()
    fig.savefig(image_path, dpi=DPI)
    plt.close(fig)
