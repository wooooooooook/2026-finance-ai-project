# 015_daily_factor_panel

014(news_horizon_decay)는 "뉴스 baseline(수익률) 대비 증분가치"를 **시장 전체 단위**로만
봤다(종목별 baseline 팩터가 없어서). 이 실험은 008/009 수준의 종목별 baseline 팩터(23개)를
**일별**로 재현해서, 뉴스가 종목별 순위(Rank IC)에 실제로 상호작용으로 기여하는지까지
본다(연구 메모 2번 검증 체크리스트의 "지평 비교"를 cross-sectional 틀로 확장하는 버전).

014와 분리한 이유: 008/009의 baseline 23개 피처 중 return/size/liquidity 계열은
`raw/daily_price`로 바로 계산되지만, 투자자별 거래·외국인 보유·일부 매크로는 **월별
데이터밖에 없어서 새로 수집해야** 한다 — 작업량이 014보다 훨씬 크다.

## 새로 수집해야 하는 것 (코드 작성 완료, 아직 실행 전)

| 항목 | 현재 상태 | 수집기 폴더 |
|---|---|---|
| 개인/외국인 일별 매수·매도 거래대금 | `raw/trading_flow`가 월별뿐 | `krx-daily-investor-flow-collector/`(종목당 매수/매도 2회 조회라 가장 오래 걸림) |
| 외국인 보유현황 일별 | `raw/trading_flow/foreign_holding`도 월별 | `krx-daily-foreign-holding-collector/` |
| 일별 시가총액·상장주식수 | **`raw/daily_price`엔 상장주식수가 없음**(종가만 있음) — log_mktcap·turnover·share_issuance 계산에 꼭 필요 | `krx-daily-marketcap-collector/` |
| 환율(fx_change), 국고채 3년/10년 금리(term_spread) | `raw/macro`가 월별, 원본 pull 보관 안 함 | `ecos-daily-macro-collector/` — 통계표 코드 확인 완료(2026-11-21), 실행만 하면 됨 |
| 기준금리(base_rate) | 월별 | `ecos-daily-macro-collector`가 환율·국고채와 함께 일별로 받음(ECOS 722Y001이 일별 주기 지원, 결정일 사이는 ECOS가 직전 값을 그대로 반복해줘서 별도 forward-fill 불필요) |
| 소비자물가(cpi_yoy) | 월별 | **일별화 불가능** — 통계청이 월 1회만 발표. 일별 모델에서는 그 달 값을 그대로 들고 가는 것 외에 방법 없음 |
| 정교한 모멘텀(mom_12_2 등 다구간 스펙) | 미구현 | 일별 가격(`raw/daily_price`)만으로 계산 가능, 014에서는 단순 1기 수익률만 썼음 — 새 수집 불필요, 피처 엔지니어링만 |

return(ret_1m 등)과 amihud_illiq는 `raw/daily_price`만으로 계산 가능. 나머지
size/liquidity(log_mktcap, turnover 등)는 위 marketcap 수집기가 필요.

4개 수집기 전부 `krx-daily-price-collector`와 같은 패턴(재시도·로그·재개 가능, 완료 후
raw/로 이동)으로 작성 완료 — 각 폴더 README의 실행법 참고. pykrx 3개(투자자 거래·외국인
보유·시가총액)는 전부 `KRX_ID`/`KRX_PW` 로그인 필요한 것으로 확인됨(2026-11-21). KRX
3개는 실행 확인 완료(2026-11-21), ECOS는 코드 작성 완료, 아직 실행 전.

## 예상 단계

1. **(코드 완료, 실행 전)** 위 4개 수집기 실행 → `raw/daily_investor_flow`,
   `raw/daily_foreign_holding`, `raw/daily_marketcap`, `raw/daily_macro`로 이동
   (`raw/daily_price`, `raw/fundamental`과 동일한 패턴)
2. 008/007과 동일한 방식으로 종목×일 baseline 팩터 패널 구성(23개 피처의 일별 버전)
3. baseline 모델 학습(xgb_rank_target 등, 006/007/013과 동일 방식) + 일별 Rank IC 확인
4. 014에서 만든 뉴스 피처(attention pool, 일별)를 가져와 009 방식(base_rank 상호작용 +
   placebo 대조군)으로 일별 증분가치 테스트
5. 008/009(월별)와 비교해 "종목별 순위 기여도" 관점에서도 지평에 따라 달라지는지 확인

## 2단계: 일별 feature 패널 (`daily_factor_panel.py`, 코드 작성 완료 2026-11-22, 아직 실행 전)

006/007(`members/wooooooooook/006_xgboost_factors`)의 23-factor 월별 baseline을 거래일
단위로 그대로 재현. 치환 매핑:

| 006(월별) | 015(일별) |
|---|---|
| `raw/returns`(adjusted_monthly_return 등) | `raw/daily_price_adjusted`(수정종가 — `daily-price-split-adjust/` 먼저 실행 필요) |
| `raw/trading_flow/price_change` | `raw/daily_marketcap`(거래대금·시가총액·상장주식수) |
| `raw/trading_flow/net_individual`·`net_foreign` | `raw/daily_investor_flow`(개인·외국인합계) |
| `raw/trading_flow/foreign_holding` | `raw/daily_foreign_holding` |
| `raw/macro`(월별, 이미 lag 적용) | `raw/daily_macro`(환율·금리) + `raw/macro`(cpi_yoy만 그 달 값 그대로 전파) |

1/3/12개월 rolling window는 거래일 수로 근사(21/63/252일 — 정확한 달력월 아님, 설계상
단순화). `foreign_own_ratio`/`foreign_limit_exhaustion`은 PREPROCESSING.md 8번(자본변동
과도기 이상치)에 따라 비정상 비율(own>200% 등)을 결측 처리. `004730`/`217590`은
PREPROCESSING.md 7번에 따라 전부 제외.

타겟은 006의 "다음 달 수익률" 대신 **다음 거래일 수익률**(014에서 뉴스 신호가 일별 지평에서
더 잘 작동한다고 본 것과 맞춤).

### 사전조건

`daily-price-split-adjust/adjust_daily_price.py`로 `raw/daily_price_adjusted/`를 먼저
만들어야 함 (액면분할·무상증자 보정, README 참고).

### RMSE 이상치 수정(2026-11-22)

거래정지(거래량=0)로 멈춰있던 가격이 재개되며 생기는 극단적 수익률(예: 008080
2013-09-11 +6,699,900%)이 train에 그대로 들어가 VAL RMSE가 MAE 대비 100배 넘게
커지는 문제가 있었음(PREPROCESSING.md 9번). 거래정지 전후 날짜와 KRX 가격제한폭(±30%)
초과 수익률을 결측 처리하도록 수정함.

## 4단계: 뉴스 증분가치 검증 (`daily_news_incremental.py`, 코드 작성 완료 2026-11-22, 아직 실행 전)

009(`members/wooooooooook/009_news_incremental`)와 동일한 방식 — 2단계 베이스라인
freeze + base_rank와 014의 일별 뉴스 attention-pool 피처(`news_feat_daily_8.csv`,
8차원)의 상호작용을 2단계 모델로 학습, 같은 구조의 랜덤노이즈 placebo(20시드 평균,
014와 동일 방식)와 비교. 009의 9-아키텍처 zoo 대신 XGBoost 하나만 사용(일별 데이터가
월별보다 20배 커서 전부 돌리는 건 비용 과다 — 014와 같은 수준의 단순화).

`daily_factor_panel.py`가 저장한 `features/panel.csv.gz`, `models/xgb_all.json`을
파일로 그대로 읽어서 쓴다(005/008/009 관례상 파이썬 import는 안 함).

### 사전조건

`daily_factor_panel.py` 실행 완료 + `014_news_horizon_decay`의
`features/news_feat_daily_8.csv` 존재.

## 실행법

**1단계(수집)**: 각 수집기 폴더의 README 참고(`krx-daily-investor-flow-collector/`,
`krx-daily-foreign-holding-collector/`, `krx-daily-marketcap-collector/`,
`ecos-daily-macro-collector/`).

**수정가격 재구성**: `daily-price-split-adjust/README.md` 참고.

**2단계(피처 패널 + 모델)**:
```bash
tmux new -s daily-factor-panel
uv run python members/wooooooooook/015_daily_factor_panel/daily_factor_panel.py | tee members/wooooooooook/015_daily_factor_panel/run.log
```
분리: `Ctrl+b d` / 재접속: `tmux attach -t daily-factor-panel`

**4단계(뉴스 증분가치)**: 2단계가 끝난 뒤 실행.
```bash
tmux new -s daily-news-incremental
uv run python members/wooooooooook/015_daily_factor_panel/daily_news_incremental.py | tee members/wooooooooook/015_daily_factor_panel/news_run.log
```
분리: `Ctrl+b d` / 재접속: `tmux attach -t daily-news-incremental`

## 결과

**1단계(데이터 수집)**: KRX 3개(투자자 거래대금·외국인 보유·시가총액) + ECOS 전부
완료(2026-11-22), `raw/`로 이동 완료.
**수정가격 재구성**: 코드 작성·검증 완료(삼성전자 2018 액면분할 등 실제 이벤트로 확인).
**2단계(일별 feature 패널 + XGBoost)**: 1차 실행 완료 후 RMSE 이상치(거래정지 재개 극단치)
발견·수정, 재실행 대기 중.
**4단계(뉴스 증분가치)**: 코드 작성 완료(2026-11-22), 2단계 재실행 후 실행 예정.
