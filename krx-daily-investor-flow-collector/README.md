# KRX 일별 투자자별 거래대금 수집

`raw/returns` 표본 1,086종목의 일별 투자자별(기관합계/기타법인/개인/외국인합계) 매수·매도
거래대금을 pykrx로 받아 `downloads/{티커}.csv`로 저장. 015_daily_factor_panel에서
개인/외국인 순매수강도·거래강도(AGENTS.md 공식)를 일별로 계산하는 데 씀.

## 왜 종목당 요청이 2번인가

pykrx `get_market_trading_value_by_date(fromdate, todate, ticker, on=...)`는
`on="매수"`/`"매도"`/`"순매수"` 중 **하나만** 돌려준다. 그런데
`개인_거래강도 = (매수+매도)/시가총액`, `개인_순매수강도 = (매수-매도)/시가총액` 공식은
매수·매도가 각각 따로 있어야 계산할 수 있어서, 종목 하나당 "매수" 조회 1번 + "매도" 조회
1번, 총 2번 요청한다. `krx-daily-price-collector`(종목당 1번)보다 요청 수가 2배라 시간도
대략 2배 걸릴 걸로 예상.

## 별도 가상환경

pykrx가 `pandas<3.0`을 요구해서 프로젝트 루트 `.venv`(pandas 3.0.6)와 충돌함.
`krx-daily-price-collector`와 동일하게 독립된 `.venv`를 둠.

```bash
cd krx-daily-investor-flow-collector
uv venv --python 3.12.13
uv pip install -r requirements.txt
```

## 저장 형식

종목코드 오름차순으로 한 종목씩 요청. 파일 하나 = 종목 하나, 전체 기간(2000-01-01~
2025-12-31) 한 번에 담김. 컬럼: `티커, 날짜, 기관합계_매수거래대금, 기타법인_매수거래대금,
개인_매수거래대금, 외국인합계_매수거래대금, 전체_매수거래대금, 기관합계_매도거래대금,
기타법인_매도거래대금, 개인_매도거래대금, 외국인합계_매도거래대금, 전체_매도거래대금`
(pykrx가 돌려주는 컬럼명 그대로 + `_매수거래대금`/`_매도거래대금` 접미사만 붙임 — raw는
가공하지 않음). AGENTS.md 공식의 "외국인"은 이 파일의 "외국인합계"에 대응.

## KRX_ID/KRX_PW 로그인 필요 (확인됨, 2026-11-21)

`krx-daily-price-collector`(가격, adjusted=False)는 로그인 없이 익명 세션으로 됐지만,
**이 투자자별 거래실적 엔드포인트는 로그인이 필요합니다** — 로그인 안 하면
`Error occurred in get_market_trading_value_and_volume_on_ticker_by_date: Expecting
value: line 1 column 1 (char 0)`로 실패함(서버가 익명 요청에 빈 응답을 줌).

`KRX_ID`/`KRX_PW` 환경변수가 설정되어 있지 않으면 스크립트 시작 시 터미널에서 직접
입력받는다(`input()`/`getpass.getpass()`) — 비밀번호를 `export`로 넘기면 쉘 히스토리에
평문으로 남는 걸 피하기 위함. 미리 환경변수로 설정해둬도 그대로 쓰고 다시 묻지 않는다.

## 주의사항

- `217590`은 `raw/trading_flow`의 KRX 원본에도 없는 종목이라 수집이 실패할 수 있음
  (`PREPROCESSING.md` 7번 참고).
- 요청 실패 시 최대 3회까지 재시도(대기 시간 2배씩 증가). 매수/매도 둘 다 성공해야 저장함
  (하나라도 실패하면 그 종목은 건너뛰고 `failed_tickers.log`에 남음, 재실행 시 처음부터
  다시 시도됨 — 둘 중 하나만 받아놓는 중간 상태를 안 둠).
- 결과물은 `raw/daily_investor_flow/{티커}.csv`에 직접 저장됨(2026-11-22부터, 이전엔
  `downloads/`에 모았다가 수동으로 옮겼음). 중단 후 재실행하면 이미 있는 종목은 건너뛰고
  이어서 받음.
- 실행 로그와 실패 종목 목록은 전부 `log/` 폴더에 쌓임.

## 실행법 (단일 계정, 순차 실행)

```bash
tmux new -s investor-flow
uv run python /home/ubuntu/2026-finance-ai-project/krx-daily-investor-flow-collector/collect_investor_flow.py
```

`KRX_ID`/`KRX_PW`를 미리 `export`해두지 않았다면 실행 즉시 터미널에서 물어본다.

로그는 스크립트가 직접 `log/collect_investor_flow.log`에도 남기므로 `tee`는 필요 없음.
분리: `Ctrl+b d` / 재접속: `tmux attach -t investor-flow`

(과거 계정 3개로 동시 실행하는 방식을 시도했으나, 3계정 동시 요청 시 KRX가 응답을
잘라 보내는 것으로 보이는 파싱 에러가 반복 발생해 2026-11-21 순차 실행으로 되돌림.
`.claim` 기반 선점 코드는 제거함.)
