# 018_feature_progression

005 → 006 → 017로 feature를 늘려 온 과정이 Rank IC를 얼마나 올렸는지, **같은 모델·같은 분할·같은
설정**에서 다시 비교한다. 005/006은 수익률 값 타깃 + VAL 단일 분할이었고 007부터 순위 타깃 + CV로 바뀌어서,
기존 README 수치끼리는 feature 효과와 타깃·평가 변경 효과가 섞여 있다.

## 고정 설정

- 모델: `xgb_rank_target` (월별 순위 타깃 XGBoost, 500트리·깊이 4·학습률 0.05·시드 42)
- 평가: walk-forward CV, val 연도 2009~2021(156개월), 매 fold 2001-01부터 직전 연도까지 학습
- test(2022~2025)는 쓰지 않음

## 비교하는 feature 구성

| 이름 | 개수 | 구성 |
|---|---|---|
| `005_step4` | 13 | 수익률 lag(`return_lag1/3/6/12`) + 개인·외국인 순매수강도·거래강도 + 매크로 5개 |
| `006` | 23 | 007 baseline (수익률·규모·유동성·수급·외국인보유·매크로) |
| `017_flowmom` | 31 | 006 + 수급 누적·가속·이상치 8개 (순매수강도 3m·12m 합·가속, 거래강도 이상치, 개인·외국인 각각) |

- 005의 `return_lagK`는 t-1월부터 거슬러 올라간 누적수익률(당월 수익률 제외)이라, 006의 `ret_1m`·`ret_3m`과 정의가 다르다.
- 005 step7(뉴스 attention)은 뉴스 임베딩 학습이 따로 필요해 제외.
- 패널은 이 폴더에서 새로 만든다(다른 실험 폴더의 코드·캐시를 가져오지 않음).

## 출력

| 파일 | 내용 |
|---|---|
| `features/panel.csv.gz` | 종목×월 패널 |
| `results/cv_monthly.csv` | feature 구성별 월별 IC·롱숏 |
| `results/summary.csv` | 구성별 CV IC, IC-IR, 적중률, 최악 연도, 롱숏 |
| `results/comparison_summary.csv` | 비교 쌍별 평균 IC 차이, 월별 paired t, 개선된 월·연도 비율 |
| `results/perm_importance_feature.csv`, `perm_importance_group.csv` | fold(연도)별 순열 중요도 |

플롯은 비교 쌍마다 폴더를 나눠 저장한다.

| 폴더 | 비교 | 파일 |
|---|---|---|
| `images/005_to_006/` | `005_step4` → `006` | `cv_ic.png`(평균 IC·95% CI·paired Δ), `cv_yearly_ic.png`(연도별 IC와 연도별 차이), `cv_monthly_delta.png`(월별 차이 156개월 + 2017년 12개월 확대), `permutation_importance_feature.png`(모델별 패널, 상위 12개, 그룹별 색), `permutation_importance_group.png` |
| `images/006_to_017/` | `006` → `017_flowmom` | 위와 같은 5개 |

## 순열 중요도

- 각 fold의 테스트 연도에서 feature(또는 그룹)를 섞고 월별 Rank IC 평균이 얼마나 떨어지는지로 계산(3회 반복 평균). 재학습 없음.
- 종목 feature는 같은 달 안에서 행을 섞고(그룹이면 같은 순열을 공유), 매크로는 달마다 값이 같아서 월 단위로 다른 달의 값과 바꿈.
- 서로 겹치는 feature(예: 개인 거래강도와 회전율)는 하나를 섞어도 다른 쪽이 대신해서 낮게 나오므로 그룹별 플롯을 함께 본다.

## 결과

아직 실행 안 함. 실행 후 `006`, `017_flowmom`의 CV IC가 017 로그(0.1055, 0.1089)와 같아야 한다.
