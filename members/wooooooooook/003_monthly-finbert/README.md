# monthly-finbert

../002_monthly-sbert/와 완전히 동일한 구조/샘플링 로직(월 전체 풀링 + 3000개/월 무작위
추출, 제목+본문 결합)을 쓰되, 임베딩 모델만 snunlp/KR-FinBert-SC(감성분류 모델,
pooler_output 사용)로 바꾼 비교용 실험.

같은 랜덤 시드(42)를 쓰므로 monthly-sbert와 정확히 같은 18만 건을 비교하게 된다 —
의미 임베딩 모델 vs 감성분류 모델 비교가 목적.

## 스크립트

- sample_and_embed.py — monthly-sbert와 동일 로직, 모델만 KR-FinBert-SC
  (AutoModel + pooler_output)로 교체. 작성만 완료, 아직 실행 안 함 (embeddings/ 비어있음)
- cluster_common.py — monthly-sbert와 동일 (겹쳐그리기/범례 버그 수정판)
- cluster_umap_monthly.py — monthly-sbert와 동일 구조의 UMAP/t-SNE/HDBSCAN
  분석 스크립트. 아직 실행 안 함 (임베딩이 없어서 돌릴 수 없음)

## 상태

실행 전. sample_and_embed.py 다음 cluster_umap_monthly.py 순서로 돌리면
monthly-sbert와 1:1 비교 가능한 결과가 나온다.
