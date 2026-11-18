# 013_shap_xgb_rank_target

007의 9개 모델 중 CV IC 1위(0.1057)였던 `xgb_rank_target`에 대해 SHAP을 뽑는 실험. 007은
모델을 저장하지 않아서, 같은 feature·하이퍼파라미터·seed로 train 분할에 다시 학습시킨다
(다른 실험 폴더를 import하지 않고 재무·거래 feature 패널 생성 코드를 이 폴더 안에 독립적으로
복사함). 뉴스를 전혀 안 써서 008/009/011/012보다 훨씬 빨리 끝난다.

재학습한 모델이 007과 같은 모델인지는 VAL Rank IC(0.1008)가 같게 나오는지로 확인한다(실행
결과 첫 줄).

SHAP은 006(xgboost_factors.py)과 동일한 방식: xgboost 내장 TreeSHAP(`pred_contribs`)이
`shap.TreeExplainer`와 같은 값을 멀티코어로 계산해준다.

이 인스턴스에 한글 폰트가 없어서(`fc-list`에 한글 지원 폰트 0개) matplotlib 그래프 텍스트가
깨지는 걸 확인함 — 그래프 안 텍스트(제목·축 라벨)는 전부 영어로 씀.

## 출력

| 단계 | 출력 |
|---|---|
| 1. feature 패널 | `features/panel.csv.gz` |
| 2. 단일 팩터 Rank IC | `results/factor_ic.csv`, `images/factor_ic.png` |
| 3. 재학습 + 재현 확인 | (콘솔 출력만) |
| 4. SHAP | `results/shap_importance_val.csv`, `images/shap_beeswarm_val.png`, `shap_group_val.png`, `shap_dependence_val.png` |

## 결과

아직 실행 안 함.
