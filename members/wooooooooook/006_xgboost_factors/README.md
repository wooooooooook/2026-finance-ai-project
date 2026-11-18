# 006_xgboost_factors

뉴스 없이 **보유한 재무·거래 데이터만으로** XGBoost를 학습하고, 각 요소(feature/feature 그룹)가
다음달 수익률 순위 예측에 어떻게 영향을 미치는지 분석하는 실험.

## 실험 목적

005에서 매크로처럼 그 달 전 종목에 같은 값이 들어가는 변수는 횡단면 순위(Rank IC)를 가르는 힘이
약했다. 006은 **종목마다 값이 다른 변수**를 늘린다. 005에서 쓰지 않았던 데이터를 추가:

- `foreign_holding`: 외국인 지분율·한도소진률
- `price_change`의 거래대금: 회전율·비유동성
- `returns`의 시가총액·주가·주식수 자체 (005는 시가총액을 분모로만 사용)
- t월 수익률 `ret_1m` (005의 `return_lag1`은 t-1월부터라 당월 수익률이 빠져 있었음)

평가 조건은 005와 동일:
- TRAIN=2001-01~2017-12, VAL=2018-01~2021-12, TEST=2022-01~2025-12
- 지표: 월별 Rank IC 평균, IC-IR, 양의 IC 비율, RMSE/MAE
- XGBoost: n_estimators=500, max_depth=4, learning_rate=0.05, seed=42
- 타겟: t+1월 수정 수익률 (순위 변환 없음)

## Feature (t월 말 기준, 7개 그룹 23개)

| 그룹 | feature | 정의 | 원본 |
|---|---|---|---|
| return | `ret_1m` | t월 수익률 | returns |
| | `ret_3m` | t-2~t월 누적수익률 | returns |
| | `mom_12_2` | t-11~t-1월 누적수익률 | returns |
| | `vol_12m` | t-11~t월 월수익률 표준편차 (최소 6개월) | returns |
| size | `log_mktcap` | log(월말 시가총액) | returns |
| | `log_price` | log(월말 실제 종가) | returns |
| | `share_issuance_12m` | log(주식수_t / 주식수_t-12) | returns |
| liquidity | `turnover_1m` | 거래대금_t / 시가총액_t | price_change |
| | `abnormal_turnover` | turnover_t / 직전 12개월 평균 turnover | price_change |
| | `amihud_illiq` | \|수익률_t\| / 거래대금_t(억원) | price_change |
| individual_flow | `individual_net_intensity`, `individual_trade_intensity` | AGENTS.md 정의 | net_individual |
| foreign_flow | `foreign_net_intensity`, `foreign_trade_intensity` | AGENTS.md 정의 (빈칸은 0으로 안 채움) | net_foreign |
| foreign_holding | `foreign_own_ratio` | 보유수량 / 상장주식수 (지분율 컬럼은 정밀도 문제로 재계산) | foreign_holding |
| | `foreign_own_chg_1m`, `foreign_own_chg_3m` | 지분율 1·3개월 변화 | foreign_holding |
| | `foreign_limit_exhaustion` | 한도소진률 | foreign_holding |
| macro | `base_rate`, `fx_change`, `term_spread`, `cpi_yoy`, `kospi_rv` | raw/macro 그대로 | macro |

- 종목 자신의 과거 값만으로 계산 (횡단면 정규화·통계 추정 없음) → test 정보 누출 없음
- `004730`, `217590` 제외 (PREPROCESSING.md 7번)
- 결측은 채우지 않고 XGBoost missing 처리에 맡김

## 단계 (`--stages`)

| 단계 | 내용 | 출력 |
|---|---|---|
| `features` | 종목×월 feature 패널 생성 (`--rebuild`로 재생성) | `features/panel.csv.gz` |
| `factor_ic` | 모델 없이 feature 하나씩의 월별 Rank IC (train/val) — 관계 방향과 시기별 부호 일관성 | `results/factor_ic.csv`, `images/factor_ic.png` |
| `train` | 23개 feature 전체로 학습 | `models/xgb_all.json` |
| `evaluate` | 전체 지표 + 연도별 IC, 005 VAL IC와 비교 | `results/monthly_ic_<split>.csv` |
| `shap` | feature별/그룹별 평균 \|SHAP\|, beeswarm, 상위 6개 dependence plot | `results/shap_importance_<split>.csv`, `images/shap_*.png` |
| `ablation` | 그룹 drop(7회)/only(6회) 재학습 → VAL IC 변화로 그룹별 실제 기여도 | `results/group_ablation.csv` |

## 실행

```bash
cd members/wooooooooook/006_xgboost_factors
python3 xgboost_factors.py
python3 xgboost_factors.py --stages features factor_ic
python3 xgboost_factors.py --stages evaluate shap --split test   # 최종 평가 때만
```

## 결과 (VAL 2018-01~2021-12)

패널: train 138,195행(월평균 677종목), val 35,210행(월평균 734종목). `foreign_*_intensity`만 커버리지 87%.

### 전체 모델

| 실험 | Rank IC | IC-IR | 양의 IC 비율 | RMSE |
|---|---|---|---|---|
| 005 step4 (매매동향+매크로 13개) | 0.0416 | 0.3855 | 66.7% | 0.1601 |
| 005 step7 (+뉴스 attention) | 0.0540 | 0.5850 | 70.8% | 0.1575 |
| **006 (23개)** | **0.0869** | **0.7844** | **85.4%** | 0.1588 |

연도별 IC: 2018 0.052 / 2019 0.096 / 2020 0.089 / 2021 0.111 — 005 walk-forward에서 음수였던
2018·2021도 양수.

월별 IC 표준편차 ≈ 0.11 → 48개월 평균의 표준오차 ≈ 0.016. 아래 ablation에서 |Δ| < 0.016은
노이즈와 구분 안 됨.

### 단일 팩터 IC (모델 없음, train / val)

| feature | train | val | 해석 |
|---|---|---|---|
| `vol_12m` | -0.084 | -0.077 | 저변동성 이상현상 |
| `turnover_1m` | -0.078 | -0.086 | 거래 과열 종목 이후 부진 |
| `individual_trade_intensity` | -0.080 | -0.078 | 〃 (turnover와 사실상 같은 정보) |
| `share_issuance_12m` | -0.040 | -0.041 | 증자 종목 부진. IC-IR -0.69/-0.60으로 가장 일관적 |
| `ret_1m` | -0.038 | -0.044 | 단기 반전 |
| `amihud_illiq` | +0.021 | +0.060 | 비유동성 프리미엄 |
| `mom_12_2` | +0.040 | **-0.035** | 부호 뒤집힘 — 모멘텀이 val 기간에 반대로 작동 |
| `log_price`, `log_mktcap` | +0.036, +0.014 | -0.010, -0.035 | 부호 뒤집힘 |
| `foreign_own_*`, `foreign_limit_exhaustion` | ≈0 | ≈0 | 신호 없음 |

### SHAP

- 매크로가 평균 |SHAP|의 53%. 하지만 매크로는 그 달 전 종목에 같은 값이라 대부분 **그 달 예측
  수준 전체를 올리고 내리는 데** 쓰이고, 순위(IC)에는 상호작용으로만 영향 → |SHAP| 크기를 Rank IC
  기여로 읽으면 안 됨. `kospi_rv` 0.042(2020-03)에서 전 종목 SHAP -0.07이 그 예
- 종목별 feature 방향은 단일 팩터 IC와 대체로 일치: 개인 거래강도↑·변동성↑·시총↑·증자↑ → 예측↓,
  저가주·비유동성↑ → 예측↑
- `mom_12_2`는 train에서 배운 대로 단조 증가(+)로 쓰이는데 val 단일 IC는 음수 → 모델 내 위험 요소

### 그룹 ablation

| 설정 | IC | Δ vs full | 설정 | IC |
|---|---|---|---|---|
| drop:return | 0.0703 | -0.0166 | only:liquidity | **0.0642** (IC-IR 0.82) |
| drop:macro | 0.0752 | -0.0116 | only:individual_flow | 0.0462 |
| drop:individual_flow | 0.0833 | -0.0035 | only:size | 0.0453 |
| drop:foreign_holding | 0.0854 | -0.0015 | only:foreign_flow | 0.0339 |
| drop:liquidity | 0.0874 | +0.0005 | only:return | 0.0108 |
| drop:size | 0.0881 | +0.0012 | only:foreign_holding | 0.0059 |
| drop:foreign_flow | 0.0890 | +0.0021 | | |

- **liquidity 단독으로 0.064**인데 빼도 IC가 안 변함 → individual_flow(거래강도)가 같은 정보를
  대체. drop 기준으로는 기여가 안 보이는 다중공선성 사례
- return은 단독 0.011로 약하지만 빼면 가장 크게 떨어짐 → 다른 feature와의 상호작용으로 기여
- 매크로를 빼면 IC 평균은 떨어지지만 IC-IR은 오름(0.78→0.92), RMSE도 개선(0.159→0.151) —
  005 step3→4와 같은 패턴. 수익률 수준을 과적합하면서 평균 IC를 올리는 구조
- foreign_holding: 단독 ≈0, 빼도 변화 없음 → 제거 후보

## 다음 후보

- walk-forward CV로 0.087이 단일 split의 우연이 아닌지 확인 (가장 우선)
- foreign_holding 제거, turnover/개인 거래강도 중복 정리 후 재측정
- 매크로 유지 여부를 CV 기준으로 결정
- 타겟을 월별 횡단면 순위로 바꾸거나 `rank:pairwise` 목적함수 사용
