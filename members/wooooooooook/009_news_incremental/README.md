# 009_news_incremental

008(news_models)은 베이스 23개 feature + 뉴스 8개를 합쳐서 9개 모델을 매번 처음부터 다시
학습했다. 그 방식은 VAL에서 거의 모든 모델이 좋아졌지만 walk-forward CV에서는 ensemble·
random_forest·extra_trees·mlp 등 대부분 사라지거나 역전됐다 — VAL 단일 분할 과적합일
가능성이 높다는 결론이었음. 이 실험은 "베이스라인 23개 feature를 완전히 freeze한 채,
뉴스가 그 위에 주는 증분 가치만" 따로 검증한다.

## 설계에서 비자명한 부분

- **뉴스는 월 단위로 전 종목에 같은 값이 붙는 feature(008과 동일)라, Rank IC는 그 달 안의
  상대 순위만 보므로 뉴스 단독으로는 이론상 절대 기여할 수 없다** — 전 종목에 같은 값을
  더해도 그 달 순위는 그대로이기 때문. 뉴스가 기여할 수 있는 유일한 통로는 종목별로 다른
  신호(베이스라인 예측)와의 상호작용뿐이다. 그래서 베이스라인을 freeze한 뒤 "뉴스 8개만"
  입력해서 잔차를 맞추는 건 항상 0이 나오는 테스트가 된다 — 베이스라인의 "그 달 예측 순위"
  (`base_rank`, 종목마다 다름) 1개를 함께 넣어서 2단계 모델이 `base_rank × 뉴스` 상호작용을
  배울 수 있게 했다. 1단계(베이스라인) 파라미터는 전혀 갱신되지 않는다.
- **placebo 대조군**: 뉴스 8차원과 똑같은 구조(월 단위 공통값)의 랜덤 노이즈 8차원을 같은
  방식으로 넣고 비교한다. "뉴스라서 좋아진 것"과 "입력을 8개 더 주면 트리 계열이 어차피
  찾아내는 가짜 개선"을 분리하는 게 목적 — `news_vs_placebo_delta`가 실제로 봐야 할 숫자.
- 2단계 모델 아키텍처는 008의 9개를 재사용하되 `xgb_reg`/`xgb_rank_target`을 `xgb` 하나로
  합쳤다. 1단계에서는 "원 수익률 vs 월별 순위"로 타겟이 달랐지만, 2단계는 둘 다 같은 잔차
  (`resid`) 하나만 예측해서 구분할 이유가 없음.
- 베이스라인 9개 아키텍처 + 랭크평균 ensemble(10개) × 2단계 8개 아키텍처 + ensemble(9개) ×
  variant(news/placebo) 전부를 VAL과 CV 13-fold 각각에 돌린다 — 조합이 많아 실행 시간이
  길다(fold당 베이스라인 9개 + 2단계 9×2×10=180개 모델 피팅).

## 출력

| 단계 | 출력 |
|---|---|
| 1~3 (feature 패널·뉴스 샘플링·임베딩) | `features/panel.csv.gz`, `news_cache/*` |
| 4. VAL 증분 평가 | `results/val_incremental_ic.csv`(월별 IC 롱포맷), `val_incremental_summary.csv` |
| 5. walk-forward CV | `results/cv_incremental_ic.csv`, `cv_incremental_summary.csv` |
| 6. 리포트 | `images/val_delta_vs_placebo.png`, `cv_delta_vs_placebo.png` |

`*_summary.csv`의 핵심 컬럼: `base_ic`(freeze 베이스라인 단독), `news_ensemble_delta`/
`placebo_ensemble_delta`(2단계 ensemble 적용 시 변화량), `news_vs_placebo_ensemble`
(뉴스가 placebo보다 얼마나 더 나은가 — 0보다 유의미하게 커야 "뉴스가 진짜 증분 정보를
준다"는 근거가 됨), `*_best_arch`/`*_best_delta`(2단계 아키텍처 중 최댓값).

## 결과

아직 실행 안 함.
