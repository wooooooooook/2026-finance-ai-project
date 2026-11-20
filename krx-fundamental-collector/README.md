# KRX 펀더멘털(PER/PBR/EPS/BPS/DIV/DPS) 수집

KOSPI 전종목의 PER, PBR, EPS, BPS, DIV(배당수익률), DPS를 **월말 거래일마다 한 번씩**
받아 `downloads/{YYYYMM}.csv`로 저장. `raw/trading_flow`와 동일한 "파일 하나 = 그 달
전종목 한 행씩" 구조.

## 왜 종목별이 아니라 월별로 받는가

처음엔 종목별로(1,086번) 받으려 했는데, `get_market_fundamental_by_ticker(날짜)`를
쓰면 **하루치 요청 한 번에 KOSPI 전종목**이 다 나온다는 걸 확인해서, 월말 거래일만
도는 방식(2000-01~2025-12, 약 312개월 → 약 312번 요청)으로 바꿨음. 요청 수가 1,086회
→ 312회로 줄어듦.

월말 거래일 날짜는 추가로 요청하지 않고, 이미 받아둔
`krx-daily-price-collector/downloads/000020.csv`의 실제 거래일에서 뽑아 씀
(pykrx 자체도 영업일 계산에 이 종목을 기준으로 씀). **그래서 이 스크립트를 돌리기
전에 `krx-daily-price-collector`로 `000020` 종목이 먼저 받아져 있어야 함.**

## 별도 가상환경

pykrx가 `pandas<3.0`을 요구해서 루트 `.venv`(pandas 3.0.6)와 충돌함. `krx-daily-price-collector`와 동일하게 독립된 `.venv`를 둠.

```bash
cd krx-fundamental-collector
uv venv --python 3.12.13
uv pip install -r requirements.txt
```

## 저장 형식

- `downloads/{YYYYMM}.csv`: 컬럼 `기준일(YYYYMMDD), 티커, BPS, PER, PBR, EPS, DIV, DPS`.
- `기준일`은 그 달의 실제 마지막 거래일(월말이 휴일이면 그 전 거래일).
- `raw/returns`의 1,086종목으로 한정하지 않고 **그 날 KOSPI에 있던 전종목**을 그대로
  저장함(trading_flow와 같은 원칙 — raw는 원본 그대로, 특정 표본으로 거르는 건 나중
  전처리 단계에서).

## 주의사항

- `krx-daily-price-collector/downloads/000020.csv`가 없으면 바로 에러로 종료됨 —
  가격 수집을 먼저 돌려서 이 파일을 만들어둘 것.
- `adjusted=False` 가격 수집처럼 KRX 로그인이 필요할 수 있음. `KRX_ID`/`KRX_PW`
  환경변수가 이미 설정돼 있다면 그대로 재사용됨.
- 요청 실패 시 최대 3회 재시도, 요청당 최대 2분 타임아웃.
- 중단 후 재실행하면 `downloads/{YYYYMM}.csv`가 이미 있는 달은 건너뛰고 이어받음.
- 로그는 `log/collect_fundamental.log`, 실패한 달은 `log/failed_months.log`.

## 실행법

```bash
export KRX_ID="본인아이디"       # 이미 설정해뒀다면 생략
export KRX_PW="본인비밀번호"

tmux new -s fundamental
cd krx-fundamental-collector
.venv/bin/python collect_fundamental.py
```

분리: `Ctrl+b d` / 재접속: `tmux attach -t fundamental`
