# 010_news_cluster_explore

001~004에서 HDBSCAN(min_cluster_size=15)을 2차원 UMAP/t-SNE 좌표 위에서 돌렸더니 클러스터가
수백 개로 쪼개졌다. 이 실험은 "클러스터 개수를 직접 통제"하는 쪽으로, 임베딩 모델 3개
(sbert/bge_m3/kpf_bert, 004와 동일, 감성분류 모델은 제외) x 클러스터링 입력 차원(4/8/16,
UMAP) x 클러스터링 방법(KMeans — k 직접 지정 / HDBSCAN — min_cluster_size로 간접 통제)을
조합해서 품질 지표와 2차원 산점도를 남긴다. 수익률 예측은 하지 않는다 — 011·012에서 쓸
클러스터링 설정(모델/차원/방법/파라미터)을 고르기 위한 탐색 실험.

## 비자명한 결정

- AgglomerativeClustering은 샘플 수(~9만 건)에서 O(n²) 메모리라 제외했다.
- HDBSCAN은 (001~004와 달리) 2차원이 아니라 4/8/16차원 UMAP 좌표에서 돌린다 — 고차원에서
  큰 클러스터가 나오는지 002~004의 결과와 비교할 수 있게 하려는 목적도 있음.
- 시각화용 2차원 UMAP은 클러스터링 입력 차원과 분리해서 모델당 1번만 계산하고 모든 조합의
  산점도에 재사용한다.

## 출력

| 단계 | 출력 |
|---|---|
| 1. 뉴스 월별 샘플링(월 300개, 전체 2001-2025) | `samples/pool.csv.gz` |
| 2. 임베딩(모델 3개) | `embeddings/{sbert,bge_m3,kpf_bert}.npz` |
| 3~4. 차원축소 + 클러스터링 | `images/{model}_dim{4,8,16}_{kmeans|hdbscan}{param}.png` |
| 5. 요약 | `clusters/sweep_metrics.csv`, `images/sweep_summary.png` |

`sweep_metrics.csv` 컬럼: model, dim, algo, param(k 또는 min_cluster_size), n_clusters,
noise_ratio, silhouette/calinski/davies(클러스터링 입력 공간 기준), fit_s.

## 결과

아직 실행 안 함.
