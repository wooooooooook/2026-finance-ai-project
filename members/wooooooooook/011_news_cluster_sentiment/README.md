# 011_news_cluster_sentiment

008/009의 뉴스 feature는 attention pool로 뭉친 8차원 벡터 하나였다. 이 실험은 기사를 의미
임베딩(기본 SBERT — 010에서 더 나은 설정을 찾으면 `SEMANTIC_MODEL`만 바꾸면 됨)으로 K=16개
클러스터로 나누고, 클러스터별로 그 달 기사들의 FinBERT 감성 점수를 평균 낸 16차원
(`clsent1~16`)을 뉴스 feature로 쓴다. "토픽별로 쪼개진 입력"이 attention-pool의 뭉친
8차원보다 비선형 모델이 상호작용을 찾기 쉬울 수 있다는 가설 검증.

평가는 009와 완전히 같은 틀(베이스라인 freeze + `base_rank` + 잔차보정 + placebo 대조군)을
그대로 복붙해서 쓴다 — 왜 이 틀이 필요한지는 009 README 참고(클러스터 감성도 뉴스처럼
"그 달 전 종목에 같은 값"이라 base_rank와의 상호작용 없이는 기여 불가능, placebo는 "16차원
이라서 생기는 capacity 증가 효과"와 "감성 신호 자체"를 구분).

## 008/009와 다른 점

- attention pool(지도학습) 자체가 없다 — KMeans는 비지도라 조기종료·epoch 로직이 필요 없음.
  split/fold가 바뀔 때마다 그 시점까지의 train 기사로만 KMeans를 다시 학습(fit)하고, 전체
  기사에 클러스터를 할당(predict)한다 — attention pool을 분할마다 재학습한 것과 같은 유출
  방지 원칙.
- FinBERT를 분류 헤드 포함으로 로드해서 긍정-부정 확률 차를 기사별 감성 점수로 쓴다
  (008/009는 분류 헤드를 버리고 임베딩만 썼음). 라벨 순서는 모델 config에서 읽어서 찾는다
  (하드코딩 안 함) — 실행 로그에 찾은 매핑이 그대로 남으니 한번 확인할 것.
- 의미 임베딩 + 감성 점수 두 번 모델을 돌려야 해서 008/009 하나보다 전처리 비용이 거의
  두 배다. 그래서 월 샘플을 1,000개가 아니라 500개로 줄였다(008/009와 완전히 같은 샘플은
  아니지만, 목적이 "클러스터 감성 입력 형태 자체"의 효과 검증이라 상관없음).

## 출력

| 단계 | 출력 |
|---|---|
| 1~2 (feature 패널·뉴스 샘플링) | `features/panel.csv.gz`, `news_cache/samples.csv.gz` |
| 3. 의미 임베딩 | `news_cache/semantic_<연도>.npz` |
| 4. 감성 점수 | `news_cache/sentiment_<연도>.npz` |
| 5. VAL 증분 평가 | `results/val_incremental_ic.csv`, `val_incremental_summary.csv` |
| 6. walk-forward CV | `results/cv_incremental_ic.csv`, `cv_incremental_summary.csv` |

`*_summary.csv` 핵심 컬럼은 009와 동일한 개념(`base_ic`, `clsent_ensemble_delta`,
`placebo_ensemble_delta`, `clsent_vs_placebo_ensemble` — 이게 0보다 유의미하게 커야 "클러스터
감성이 placebo보다 낫다"는 근거).

## 결과

아직 실행 안 함.
