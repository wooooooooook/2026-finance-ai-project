# sbert

제목만 사용해 snunlp/KR-SBERT-V40K-klueNLI-augSTS(범용 한국어 문장 의미 임베딩 모델)로
임베딩하는 1차 실험. 감성분류 모델(KR-FinBert-SC) 대신 의미 유사도 모델을 써보기 위해 만듦.

## 데이터

- bigkinds-crawler/downloads의 매월 15일자 파일만 사용 (60개, 2001~2005년)
- 월 전체가 아니라 하루치만 대표로 뽑은 초기 방식

## 스크립트

- embed_titles_sbert.py — 제목만 추출해 KR-SBERT로 임베딩, embeddings/*.npz로 저장. 실행 완료
- embed_titlebody_sbert.py — 제목+본문 결합 버전. 작성만 하고 실행 안 함 — 이후
  "월 전체 데이터를 풀링해서 샘플링하자"는 방향으로 바뀌면서 ../002_monthly-sbert/로 대체됨
- cluster_common.py — UMAP/t-SNE 플로팅 + 실루엣 등 클러스터 품질 지표 유틸
- sample_cluster_umap_sbert.py — 연도별 1000개씩 무작위 샘플링(총 5000건) 후 UMAP과
  t-SNE 각각으로 2차원 축소, HDBSCAN 클러스터링. 연도별/클러스터별 색상 이미지 4장 +
  CSV + 품질 지표 출력. 실행 완료

## 상태

1차 실험 단계. 여기서 제목만으로는 토픽 클러스터가 잘 안 나뉜다는 걸 확인했고,
이후 ../002_monthly-sbert/에서 제목+본문 결합 + 월 전체 풀링 방식으로 개선함.
