"""step3_cluster.py에서 쓰는 클러스터 품질 지표 유틸."""

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def print_cluster_metrics(embeddings: np.ndarray, coords: np.ndarray, labels: np.ndarray, method_name: str) -> None:
    mask = labels != -1
    n_labels = len(set(labels[mask]))

    if n_labels < 2 or mask.sum() < 2:
        print("클러스터 품질 지표 계산 불가 (유효 클러스터가 2개 미만)")
        return

    print(f"클러스터 품질 지표 (노이즈 {(~mask).sum()}개 제외, {mask.sum()}개 대상, {n_labels}개 클러스터)")
    for space_name, space in [("원본 임베딩", embeddings), (f"{method_name} 2차원 좌표", coords)]:
        sil = silhouette_score(space[mask], labels[mask])
        ch = calinski_harabasz_score(space[mask], labels[mask])
        db = davies_bouldin_score(space[mask], labels[mask])
        print(
            f"  [{space_name}] silhouette={sil:.4f} | calinski-harabasz={ch:.1f} | davies-bouldin={db:.4f}"
        )
