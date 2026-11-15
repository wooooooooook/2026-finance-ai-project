# yearly-multi-model

뉴스 제목+본문 임베딩 모델 3개를 같은 샘플로 비교하는 4번째 실험. 002/003과 달리
월 단위가 아니라 연도 단위로 2001~2020년 전체를 다룬다.

## 데이터

- raw/news의 일별 csv.gz 전체를 연도 단위로 풀링
- 연도별로 5000개씩 무작위 추출 (고정 시드), 2001~2020년 20개 연도 기준 총 10만 건
- 제목과 본문(최대 200자)을 제목. 본문 형태로 합쳐서 임베딩
- 세 모델 모두 같은 샘플(step1에서 한 번만 생성한 캐시)을 공유해서 비교함

## 비교 모델

- sbert: snunlp/KR-SBERT-V40K-klueNLI-augSTS. 범용 한국어 문장 의미 임베딩, 001/002에서 이미 사용
- bge_m3: upskyy/bge-m3-korean. large급(약 5.6억 파라미터), 범용 고성능, 최대 8192토큰 지원하지만
  지금 입력 길이(150~250토큰)엔 과함. 실측 처리량 40.7건/초로 base급보다 4배 이상 느림
- kpf_bert: leewaay/kpf-bert-base-klueNLI-klueSTS-MSL512. 한국언론진흥재단 뉴스 코퍼스로
  프리트레이닝된 뒤 KLUE-NLI/STS로 파인튜닝됨 - 뉴스 도메인과 STS 학습을 둘 다 만족하는 후보

## 스크립트

- step1_sample_pool.py — raw/news를 연도별로 풀링해 5000개씩 샘플링, samples/sampled_pool.csv.gz로
  저장. 세 모델이 공유하는 캐시라 한 번만 실행하면 됨
- embed_common.py — step2 스크립트들이 공유하는 임베딩 실행 함수 (라이브러리, 직접 실행 안 함)
- step2_embed_sbert.py / step2_embed_bge_m3.py / step2_embed_kpf_bert.py — 캐시된 샘플을
  각 모델로 임베딩, embeddings/{key}_embeddings.npz로 저장. 모델마다 독립 실행 가능
- cluster_common.py — 클러스터 품질 지표(실루엣 등) 유틸 (라이브러리)
- step3_cluster.py — 임베딩마다 UMAP과 t-SNE 각각 2차원 축소 + HDBSCAN 클러스터링,
  좌표와 라벨을 clusters/{key}_clusters.csv에 저장. 이미지는 안 그림
- plot_common.py — 플로팅 유틸 (라이브러리). 10만 건을 그대로 찍으면 겹쳐서 안 보이므로
  점을 2만 개로 무작위 서브샘플링해서 그리고, 투명도와 고해상도(dpi 600)를 적용함
- step4_plot.py — clusters/*.csv만 읽어서 이미지를 그림 (재계산 없음). 모델 3개 ×
  (UMAP/t-SNE) × (연도/클러스터) = 총 12장

## 실행 순서

step1_sample_pool.py 한 번 실행 후, step2 세 스크립트를 각각 실행, 그다음
step3_cluster.py, 마지막으로 step4_plot.py. 그림만 다시 그리고 싶을 때는
step4_plot.py만 다시 실행하면 된다 (step3의 계산 결과가 CSV로 남아있어서 재계산 안 함).

## 예상 소요 시간 (이 노트북 기준 실측 외삽)

- step1 (raw/news 약 7300개 파일 읽기): 약 5~6분
- step2 sbert: 10만 건 기준 약 10분 (실측 171.5건/초)
- step2 bge_m3: 10만 건 기준 약 41분 (실측 40.7건/초)
- step2 kpf_bert: 아직 미측정, sbert와 같은 base급이라 비슷한 수준으로 추정
- step3 (UMAP+t-SNE+HDBSCAN, 모델당): t-SNE 이웃 탐색이 고차원에서 거의 제곱에 가깝게
  느려지는 경향이 있어 10만 건 기준 모델당 대략 5~10분으로 추정 (180,000건 기준 실측
  대비 외삽, 정확하지 않음)
- step4: CSV만 읽으므로 수 초~수십 초

## 상태

작성만 완료, 아직 실행 안 함.
