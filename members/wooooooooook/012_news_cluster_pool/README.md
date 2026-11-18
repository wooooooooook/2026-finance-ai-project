# 012_news_cluster_pool

008/009의 attention pool은 "이 기사가 중요한지"를 기사 단위 점수 MLP로 지도학습해서 가중합
했다. 이 실험은 그 점수 학습 부분을 비지도 클러스터 비율로 완전히 대체한다: 기사를 의미
임베딩(기본 SBERT — 010 결과로 `SEMANTIC_MODEL`만 바꿀 수 있음)으로 K=16개 클러스터로
나누고, 그 달 기사들이 클러스터에 분포된 비율로 클러스터 중심(centroid)들을 가중합해서
`pooled_그달 = Σ_c 비율_c × centroid_c`를 만든다. 이 pooled 벡터를 008/009와 동일하게
projection(→8차원)해서 다음 달 평균 수익률을 예측하도록 학습시키고, 그 8차원(`clpool1~8`)을
최종 feature로 쓴다 — 005 step9(attention 없는 단순 평균)의 클러스터 버전.

평가는 009와 완전히 같은 틀(베이스라인 freeze + `base_rank` + 잔차보정 + placebo 대조군)을
그대로 쓴다. 자세한 이유는 009 README 참고.

## 008/009와 다른 점

- pooled 벡터가 "클러스터 비율 × centroid"로 이미 고정돼 있어서, 매 epoch마다 기사 임베딩을
  순회할 필요가 없다 — 009의 attention pool보다 이 단계 학습이 훨씬 가볍다(학습 대상은
  projection+head뿐).
- KMeans는 split/fold가 바뀔 때마다 그 시점까지의 train 기사로만 다시 학습(fit)하고, 전체
  기사에 클러스터를 할당(predict)한다 — attention pool 재학습과 같은 유출 방지 원칙.
- 감성분류(FinBERT)를 전혀 안 쓴다 — 대표값이 임베딩 그 자체라 011과 달리 의미 임베딩
  1회만 돌리면 된다(011보다 전처리 비용이 가볍음).

## 출력

| 단계 | 출력 |
|---|---|
| 1~2 (feature 패널·뉴스 샘플링) | `features/panel.csv.gz`, `news_cache/samples.csv.gz` |
| 3. 의미 임베딩 | `news_cache/semantic_<연도>.npz` |
| 4. VAL 증분 평가 | `results/val_incremental_ic.csv`, `val_incremental_summary.csv` |
| 5. walk-forward CV | `results/cv_incremental_ic.csv`, `cv_incremental_summary.csv` |

`*_summary.csv` 핵심 컬럼은 009/011과 동일한 개념(`base_ic`, `clpool_ensemble_delta`,
`placebo_ensemble_delta`, `clpool_vs_placebo_ensemble`).

## 결과

아직 실행 안 함.
