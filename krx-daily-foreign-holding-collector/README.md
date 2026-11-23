# KRX 일별 외국인 보유현황 수집

`raw/returns` 표본 1,086종목의 일별 외국인 보유현황을 pykrx로 받아 `downloads/{티커}.csv`로
저장. `raw/trading_flow/foreign_holding`(월별, 그 달 마지막 거래일 기준 스냅샷)의 일별
버전. 015_daily_factor_panel의 foreign_own_ratio·foreign_own_chg_1m/3m·
foreign_limit_exhaustion 피처에 씀.

## 별도 가상환경

pykrx가 `pandas<3.0`을 요구해서 프로젝트 루트 `.venv`(pandas 3.0.6)와 충돌함.
`krx-daily-price-collector`와 동일하게 독립된 `.venv`를 둠.

```bash
cd krx-daily-foreign-holding-collector
uv venv --python 3.12.13
uv pip install -r requirements.txt
```

## 저장 형식

종목코드 오름차순으로 한 종목씩 요청(`get_exhaustion_rates_of_foreign_investment_by_date`).
파일 하나 = 종목 하나, 전체 기간(2000-01-01~2025-12-31) 한 번에 담김. 컬럼: `티커, 날짜,
상장주식수, 보유수량, 지분율, 한도수량, 한도소진률`(pykrx 반환 그대로).

`raw/trading_flow/foreign_holding` README에 적힌 대로 "지분율" 컬럼은 pykrx 자체 정밀도가
낮으니, 더 정확한 값이 필요하면 `보유수량 ÷ 상장주식수 × 100`으로 직접 재계산할 것.

## KRX_ID/KRX_PW 로그인 필요할 수 있음

`krx-daily-investor-flow-collector`(투자자별 거래실적)에서 로그인 없이 돌렸더니 서버가
익명 요청을 거부해서 실패하는 게 확인됨(2026-11-21). 이 엔드포인트도 같은 "상세 데이터"
카테고리라 동일하게 로그인이 필요할 가능성이 있음.

`KRX_ID`/`KRX_PW` 환경변수가 설정되어 있지 않으면 스크립트 시작 시 터미널에서 직접
입력받는다(`input()`/`getpass.getpass()`) — 비밀번호를 `export`로 넘기면 쉘 히스토리에
평문으로 남는 걸 피하기 위함. 미리 환경변수로 설정해둬도 그대로 쓰고 다시 묻지 않는다.

## 주의사항

- `217590`은 `raw/trading_flow`의 KRX 원본에도 없는 종목이라 수집이 실패할 수 있음
  (`PREPROCESSING.md` 7번 참고).
- 요청 실패 시 최대 3회까지 재시도(대기 시간 2배씩 증가).
- 결과물은 `raw/daily_foreign_holding/{티커}.csv`에 직접 저장됨(2026-11-22부터, 이전엔
  `downloads/`에 모았다가 수동으로 옮겼음). 중단 후 재실행하면 이미 있는 종목은 건너뛰고
  이어서 받음.
- 실행 로그와 실패 종목 목록은 전부 `log/` 폴더에 쌓임.

## 실행법 (단일 계정, 순차 실행)

```bash
tmux new -s foreign-holding
uv run python /home/ubuntu/2026-finance-ai-project/krx-daily-foreign-holding-collector/collect_foreign_holding.py
```

`KRX_ID`/`KRX_PW`를 미리 `export`해두지 않았다면 실행 즉시 터미널에서 물어본다.

로그는 스크립트가 직접 `log/` 폴더에도 남기므로 `tee`는 필요 없음.
분리: `Ctrl+b d` / 재접속: `tmux attach -t foreign-holding`
