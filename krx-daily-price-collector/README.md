# KRX 일별 가격 수집

`raw/returns` 표본 1,086종목의 일별 OHLCV를 pykrx로 받아 `downloads/{티커}.csv`로 저장.
목적: 월별 뉴스 신호 대 일별 뉴스 신호 비교 실험(뉴스 지평 희석 검증)에 쓸 일별 수익률 확보.

## 별도 가상환경을 쓰는 이유

pykrx 최신 버전이 `pandas<3.0`을 요구해서 프로젝트 루트 `.venv`(pandas 3.0.6)와 충돌함.
그래서 이 폴더 안에 독립된 `.venv`를 둠.

```bash
cd krx-daily-price-collector
uv venv --python 3.12.13
uv pip install -r requirements.txt
```

## 저장 형식

종목코드 오름차순으로 한 종목씩 요청. 파일 하나 = 종목 하나, 전체 기간(2000-01-01~2025-12-31)
한 번에 담김. 컬럼: `티커, 날짜, 시가, 고가, 저가, 종가, 거래량, 거래대금, 등락률`
(pykrx `get_market_ohlcv_by_date`가 반환하는 그대로 + 티커 컬럼만 추가).

## 왜 `adjusted=False`인가

처음에는 `adjusted=True`(네이버 소스)로 받았는데, 이 경로는 fromdate를 아무리 과거로
줘도 서버에 todate를 보내지 않고 **"오늘 기준 최근 N일"만 요청하는 방식**이라서, 네이버
서버가 응답을 최근 약 11년치(2026-11 기준 2014년 이후)로 잘라버리는 걸 실측으로 확인함.
날짜를 구간으로 나눠 요청해도 매번 "오늘 기준 최근 11년" 범위만 받고 필터링만 달라질
뿐이라 과거 데이터를 받을 방법이 없었음.

`adjusted=False`는 KRX 소스를 쓰며 fromdate/todate를 실제로 서버에 보내는 날짜범위
조회라 이 문제가 없음. 대신:
- **액면분할·무상증자 등 자본변동이 반영되지 않은 raw 종가.** 분할 시점에 가격이 뚝
  끊겨 보일 수 있음 — 나중에 일별 수익률 계산 시 분할 이벤트를 따로 처리해야 할 수 있음.
- 배당도 당연히 반영 안 됨. `raw/returns`의 `adjusted_monthly_return`(배당까지 반영)과는
  다른 방법론이므로 직접 비교 전 상관 체크가 필요함.
- KRX_ID/KRX_PW 로그인은 필요 없음(없으면 익명 세션으로 조회, pykrx 소스 확인).

## 주의사항

- `217590`은 `raw/trading_flow`의 KRX 원본에도 없는 종목이라 수집이 실패할 수 있음
  (`PREPROCESSING.md` 7번 참고).
- 요청 실패 시 최대 3회까지 재시도(대기 시간 2배씩 증가).
- 중단 후 재실행하면 `downloads/{티커}.csv`가 이미 있는 종목은 건너뛰고 이어서 받음.
- 실행 로그와 실패 종목 목록은 전부 `log/` 폴더에 쌓임
  (`log/collect_daily_price.log`, `log/failed_tickers.log`).

## 실행법

```bash
tmux new -s daily-price
.venv/bin/python collect_daily_price.py
```

로그는 스크립트가 직접 `log/` 폴더에 파일로 남기므로 `tee`는 필요 없음.
분리: `Ctrl+b d` / 재접속: `tmux attach -t daily-price`
