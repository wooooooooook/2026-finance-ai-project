"""
clusters/{key}_clusters.csv를 읽어서 모델별로 이미지 4장(UMAP-연도, UMAP-클러스터,
t-SNE-연도, t-SNE-클러스터)을 그린다. 3개 모델 기준 총 12장.

재계산 없이 CSV만 읽어서 그리므로 빠르다 - 그림만 다시 그리고 싶을 때는 이 스크립트만
다시 돌리면 된다 (step3_cluster.py를 다시 돌릴 필요 없음).

사용법:
    cd 004_yearly_multi_model
    uv run python step4_plot.py
"""

from pathlib import Path

import pandas as pd

from plot_common import plot_by_category

BASE_DIR = Path(__file__).resolve().parent
CLUSTERS_DIR = BASE_DIR / "clusters"
IMAGES_DIR = BASE_DIR / "images"

MODELS = ["sbert", "bge_m3", "kpf_bert"]


def plot_model(key: str) -> None:
    csv_path = CLUSTERS_DIR / f"{key}_clusters.csv"
    if not csv_path.exists():
        print(f"[{key}] 클러스터링 결과 없음, 건너뜀: {csv_path}")
        return

    print(f"=== {key} ===")
    df = pd.read_csv(csv_path)
    years = df["year"].to_numpy()

    for method, x_col, y_col, cluster_col in [
        ("UMAP", "umap_x", "umap_y", "umap_cluster"),
        ("tSNE", "tsne_x", "tsne_y", "tsne_cluster"),
    ]:
        coords = df[[x_col, y_col]].to_numpy()
        labels = df[cluster_col].to_numpy()
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        plot_by_category(
            coords,
            years,
            f"{method} projection ({key}) colored by year (n={len(df)}, clusters={n_clusters})",
            IMAGES_DIR / f"{key}_{method.lower()}_year.png",
            legend_title="year",
            method_name=method,
        )
        plot_by_category(
            coords,
            labels,
            f"{method} + HDBSCAN clustering ({key}) (n={len(df)}, clusters={n_clusters})",
            IMAGES_DIR / f"{key}_{method.lower()}_cluster.png",
            legend_title="cluster",
            noise_value=-1,
            method_name=method,
        )
        print(f"  [{method}] 이미지 2장 저장 완료")


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for key in MODELS:
        plot_model(key)


if __name__ == "__main__":
    main()
