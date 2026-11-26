# 017_flow_momentum

007의 재무·거래 feature 23개(base)에 투자자 유형별(개인·외국인) 수급 모멘텀 8개를 더한 버전(flowmom)이
Rank IC를 개선하는지 같은 모델·같은 분할(VAL + walk-forward CV 156개월)로 비교한다.

## 추가 feature (유형별 4개 × 개인·외국인 = 8개)

월별 순매수강도 `net_t` = (매수−매도 거래대금)/시가총액, 거래강도 `trade_t` = (매수+매도)/시가총액(AGENTS.md 정의).

| feature | 정의 |
|---|---|
| `{유형}_net_3m` | `net` 최근 3개월 합 |
| `{유형}_net_12m` | `net` 최근 12개월 합 |
| `{유형}_net_accel` | `net_t` − 직전 3개월(t-3~t-1) `net` 평균 |
| `{유형}_trade_abnormal` | `trade_t` / 직전 12개월 `trade` 평균(최소 6개월) |

- 외국인 순매수 결측(2013년 이전 소형주 등)은 0으로 채우지 않고, 롤링 윈도우에 결측이 있으면 NaN
- 모멘텀을 "투자자별 순매수의 누적·가속"으로 정의 — 007의 가격 모멘텀(`mom_12_2`)과 별개

## 모델

`xgb_rank_target`(월별 순위 타깃 XGBoost) 하나만, 시드 5개(42~46)로 학습. `seed_avg`는 5개 시드의
월별 순위 평균 — 시드 노이즈를 줄인 최종 비교 대상. 시드별 IC는 모델명 `xgb_rank_target_s{시드}`로 남는다.
base 시드 42의 VAL IC가 007 로그의 0.1008과 같게 나오는지로 재현을 확인한다.

## 출력

| 단계 | 출력 |
|---|---|
| feature 패널 | `features/panel.csv.gz` |
| VAL | `results/val_metrics.csv`, `val_monthly_ic.csv` |
| CV | `results/cv_monthly.csv` |
| 요약 | `results/summary.csv`(모델별 Δ·월별 paired t·개선 월 비율), `images/cv_yearly_ic_delta.png` |

## 결과

아직 실행 안 함.
