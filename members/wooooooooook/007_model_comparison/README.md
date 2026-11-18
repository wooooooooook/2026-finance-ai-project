# 007_model_comparison

006과 같은 재무·거래 feature 23개(뉴스 없음, 정의는 006 README 참고)로 여러 모델을 학습해 비교하고,
모델별로 각 팩터가 Rank IC에 얼마나 기여하는지 측정하는 실험.

006은 VAL 48개월 단일 분할에서 IC 0.087이 나왔지만 평균의 표준오차가 ≈0.016이라 모델 간 차이를
가리기 어렵다. 그래서 단일 분할과 함께 walk-forward CV(13개 연도, 156개월)로도 비교한다.

## 모델

| 모델 | 입력 | 타겟 | 설정 |
|---|---|---|---|
| `ridge` | 월별 순위 정규화 | 월별 순위 | alpha=1.0 |
| `mlp` | 월별 순위 정규화 | 월별 순위 | (64, 32), early stopping |
| `random_forest` | 원본 (결측=train 중앙값) | 수익률 | 300 trees, min_samples_leaf=100, max_features=0.5 |
| `extra_trees` | 〃 | 수익률 | 〃 |
| `xgb_reg` | 원본 | 수익률 | 006과 동일 (500, depth 4, lr 0.05) |
| `xgb_rank_target` | 원본 | 월별 순위 | 〃 |
| `xgb_pairwise` | 원본 | 월별 수익률 10분위 | `rank:pairwise`, 월 = query |
| `lgbm_reg` | 원본 | 수익률 | 500, depth 4, num_leaves 15, lr 0.05 |
| `catboost_reg` | 원본 | 수익률 | 500, depth 4, lr 0.05 |
| `ensemble` | — | — | 위 9개 예측의 월별 순위 평균 |

- 월별 순위 정규화: 종목 feature는 그 달 안에서 순위(-0.5~0.5, 결측=0), 매크로는 train 평균·표준편차로 표준화.
  선형 모델에서 매크로는 그 달 전 종목에 같은 값을 더할 뿐이라 순위에 영향 없음
- 월별 순위 타겟: 극단 수익률 종목에 학습이 끌려가지 않게 하고 Rank IC에 맞춤

## 평가 지표

- Rank IC 평균, IC-IR, IC>0 비율 (005·006과 동일)
- L-S: 매월 예측 상위 10% 평균 수익률 − 하위 10% 평균 수익률 (동일가중)
- CV worst yr: walk-forward 13개 연도 중 연평균 IC 최저값
- CV 지표는 fold 평균이 아니라 156개월 전체를 모아서 계산 (005 step5에서 fold 길이 차이로 평균이 왜곡된 문제 회피)

## 팩터 기여도 (permutation importance, VAL)

train으로 학습한 각 모델에 대해 VAL에서 feature(또는 그룹)를 섞었을 때 Rank IC가 얼마나 떨어지는지(ΔIC).
3회 반복 평균. 모델 종류와 무관하게 같은 기준으로 비교 가능.

- 매크로는 "그 달 전 종목에 같은 값" 구조를 유지하도록 월 단위로 섞음
- 그룹 단위는 그룹 내 feature를 같은 순서로 함께 섞음 → 그룹 안에서 상관 높은 feature끼리 서로 대체해
  기여가 안 보이는 문제(006 ablation의 liquidity 사례)를 완화
- ΔIC < 0이면 섞었을 때 오히려 IC가 오름 → 그 모델에서 해당 팩터가 노이즈로 작용

## 실행 순서와 출력

| 순서 | 출력 |
|---|---|
| 1. feature 패널 (있으면 재사용) | `features/panel.csv.gz` |
| 2. VAL 평가 + permutation importance | `results/val_metrics.csv`, `val_monthly_ic.csv`, `importance.csv` |
| 3. walk-forward CV | `results/cv_monthly.csv` (모델 × 월 IC·L-S) |
| 4. 리포트 | `results/summary.csv`, `images/model_comparison.png`, `cv_yearly_ic.png`, `importance_feature.png`, `importance_group.png` |

## 결과

아직 실행 안 함.
