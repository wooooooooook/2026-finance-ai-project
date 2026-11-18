# 008_news_models

007의 모델 9개(+앙상블)에 경제일간지 뉴스 feature를 추가했을 때 성능이 얼마나 바뀌는지 비교하고,
뉴스 추가 모델에서 각 팩터의 기여도를 측정하는 실험.

raw/news는 기업명 검색이 아니라 BigKinds 경제일간지 카테고리 전체를 수집한 것이라 기업별 뉴스가
아니다(raw/news README). 그래서 기업 뉴스는 시도하지 않고, 경제일간지 전체를 쓴다.

## 뉴스 feature: attention pooling (005 step7 방식)

005의 뉴스 실험 4가지(step6 선형 날짜가중 평균, step7 FinBERT attention pooling, step8 BGE-M3
key/FinBERT value attention, step9 attention 없는 mean pooling) 중 step7이 전 지표에서 가장
좋았다. PCA 같은 비지도 압축 대신, "이 기사가 다음 달 수익률 예측에 얼마나 중요한지"를 작은 MLP로
직접 학습시키는 방식이다.

- 샘플링: 005 step6과 동일, 일자별 기사 수 비중대로 월 1,000개 (PREPROCESSING.md 1~4·6번 전처리 적용)
- 임베딩: 기사 단위 FinBERT(KR-FinBert-SC) [CLS] 임베딩(768차원), 파라미터 고정
- 집계: score MLP(768→32→1) → 그 달 안에서 softmax → 가중합 → 투영(768→8) = 최종 feature `news_att1~8`
- 학습 신호: 투영 벡터로 "다음 달 평균 수익률"을 예측(MSE). 투영 자체가 종목 무관(그 달 전 종목에
  같은 값)이라 순위를 직접 배우지는 않는다 — 기업 뉴스가 아니므로 이 한계는 구조적으로 불가피
- 유출 방지: 분할이 바뀔 때마다(메인 1회 + CV 13-fold) 그 시점까지의 train만으로 attention pool을
  다시 학습. CV fold는 그 fold의 test 연도를 들여다보지 않도록 조기종료 없이 메인 분할에서 정한
  epoch 수를 고정 사용

## 비교 구성

007과 같은 모델 9개 + 앙상블을, feature 두 버전으로 각각 학습해서 비교한다.
- `base`: 007과 동일한 재무·거래 feature 23개
- `news`: base + `news_att1~8`

평가는 VAL(2018-2021) 단일 분할과 walk-forward CV(2009-2021, 156개월) 둘 다 쓴다. 지표는 Rank IC,
IC-IR, IC>0 비율, L-S 스프레드(상위 10% − 하위 10% 월별 평균 수익률)에 더해 **score**(005 README의
채점 기준 — 월별 IC 0 이하 0점, 0.06 이상 5점, 사이는 선형 — 를 CV 156개월에 적용한 평균)를 추가했다.

## 팩터 기여도 (permutation importance)

news 버전 모델에 대해, VAL에서 feature(또는 그룹)를 섞었을 때 Rank IC가 얼마나 떨어지는지(ΔIC,
3회 반복 평균). 뉴스(`news_att*`)는 매크로와 같은 월 단위 feature라 월 단위로 섞는다. 그룹은 그룹
내 feature를 같은 순서로 함께 섞어 그룹 안 상관으로 인한 과소평가를 줄인다.

## 실행 순서와 출력

| 순서 | 출력 |
|---|---|
| 1. 재무·거래 feature 패널 | `features/panel.csv.gz` |
| 2. 뉴스 월별 샘플링 | `news_cache/samples.csv.gz`, `month_counts.csv` |
| 3. FinBERT 기사 임베딩 | `news_cache/emb_<연도>.npz` |
| 4. attention pool 학습 + VAL 평가 + permutation importance | `results/val_metrics.csv`, `val_monthly_ic.csv`, `importance.csv` |
| 5. walk-forward CV | `results/cv_monthly.csv` |
| 6. 리포트 | `results/summary.csv`, `images/model_comparison.png`, `cv_yearly_ic_delta.png`, `importance_feature.png`, `importance_group.png` |

## 결과

아직 실행 안 함.
