# ECOS 일별 매크로(환율·국고채 금리·기준금리) 수집

환율(원/달러)·국고채 3년/10년 금리·기준금리를 한국은행 ECOS OpenAPI로 일별 수집해
`raw/daily_macro/{시리즈명}.csv`로 저장. 015_daily_factor_panel의 fx_change·term_spread·
base_rate 일별 버전에 씀.

## 통계표/항목 코드 (2026-11-21 ECOS 사이트에서 직접 확인함)

| 시리즈 | 통계표코드 | 항목코드 | 비고 |
|---|---|---|---|
| fx_usd_krw | 731Y003 | 0000003 | 원/달러(종가 15:30) — 0000013("종가")은 2024-07-01부터만 데이터가 있어서 더 오래 쌓인 이 항목으로 교체함 |
| treasury_3y | 817Y002 | 010200000 | 국고채(3년), 시장금리(일별) 표 |
| treasury_10y | 817Y002 | 010210000 | 국고채(10년), 같은 표 |
| base_rate | 722Y001 | 0101000 | 한국은행 기준금리 |

## base_rate를 그냥 똑같이 받기로 함

원래는 "금통위 결정일만 알면 된다"고 별도 수집 안 하려 했는데, 통계표 722Y001이 일별(D)
주기도 지원해서 그냥 나머지와 동일하게 받기로 함 — 결정일 사이는 ECOS가 알아서 직전 값을
반복해주므로 별도 forward-fill 로직이 필요 없음.

## cpi_yoy는 왜 여기 없는가

소비자물가는 통계청이 월 1회만 발표해서 구조적으로 일별화 불가능 — 일별 모델에서는 그 달
값을 그대로 들고 가는 수밖에 없음(`raw/macro` README 참고).

## 가상환경

pykrx를 안 써서(requests만 사용) 프로젝트 루트 `.venv`로 바로 실행 가능 — 별도 가상환경
불필요.

## ECOS_API_KEY 필요

[ecos.bok.or.kr](https://ecos.bok.or.kr)에서 무료로 발급받은 인증키를 환경변수로 설정:

```bash
export ECOS_API_KEY="본인_인증키"
```

## 저장 형식

`raw/daily_macro/{시리즈명}.csv`(fx_usd_krw.csv, treasury_3y.csv, treasury_10y.csv,
base_rate.csv) — ECOS `StatisticSearch` API가 돌려주는 JSON 행을 그대로 DataFrame으로
저장(가공 안 함).

## 주의사항

- 결과물은 `raw/daily_macro/`에 직접 저장됨(2026-11-22부터, 이전엔 `downloads/`에
  모았다가 수동으로 옮겼음). 중단 후 재실행하면 이미 받은 시리즈(파일 존재)는 건너뜀.

## 실행법

```bash
export ECOS_API_KEY="본인_인증키"

tmux new -s ecos-macro
uv run python ecos-daily-macro-collector/collect_ecos_macro.py 2>&1 | tee -a ecos-daily-macro-collector/log/run.log
```

분리: `Ctrl+b d` / 재접속: `tmux attach -t ecos-macro`
