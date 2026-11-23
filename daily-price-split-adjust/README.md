# daily_price 수정가격 재구성

`raw/daily_price`(수정 안 한 종가, `adjusted=False`)를 `raw/returns`의
`adjusted_monthly_return`과 같은 방식(액면분할·무상증자 등 자본변동 반영)으로 보정해서
`raw/daily_price_adjusted/{티커}.csv`에 저장. `raw/daily_price` 원본은 건드리지 않음.

## 왜 필요한가

`raw/daily_price`는 KRX 소스 그대로라 분할·무상증자 시점에 가격이 뚝 끊겨 보임(`raw/daily_price`
README 참고). 일별 수익률을 월별 수익률(`returns`)과 같은 방법론으로 계산하려면 이 끊김을
보정해야 함.

## 방법

`raw/daily_price` 자체엔 분할 비율 정보가 없어서, `raw/daily_marketcap`의 **상장주식수**
일별 변화로 분할/무상증자 이벤트를 역산한다.

1. 상장주식수가 하루 사이 `SHARE_JUMP_THRESHOLD`(기본 5%) 이상 변하고,
2. 같은 날 종가가 그 비율의 **역수만큼 반대로** 움직이면(오차 `PRICE_INVERSE_TOLERANCE`
   10% 이내) — 기계적 분할/무상증자로 판단.
3. 상장주식수는 늘었는데 종가가 그만큼 안 움직이면(실제 자금이 들어온 유상증자로 추정) —
   보정하지 않음. 유상증자는 가격을 인위적으로 깨지 않기 때문.

이벤트가 감지되면 그 이벤트 **이전** 모든 날짜의 가격에 `1/상장주식수비율`을 누적으로 곱해서
최신 날짜(raw 그대로, 미보정) 기준 스케일로 역산한다(back-adjustment, `returns`의 수정주가와
동일한 사상).

**거래량·거래대금·등락률(원본)은 보정하지 않음** — 분할은 가격×주식수를 그대로 두고 비율만
바꾸는 이벤트라, 거래대금(원화) 같은 원화 단위 값은 분할에 영향을 안 받음 (시가총액·거래강도
공식도 전부 원화 단위라 마찬가지 — 대화 중 직접 확인함).

## 알려진 한계

- 상장주식수 변화만으로 유상증자/무상증자/분할을 완벽히 구분할 수 없음. 종가 역방향 변화
  조건으로 유상증자는 대부분 걸러지지만, 100% 보장은 아님.
- 상장주식수 데이터 자체의 노이즈(예: `PREPROCESSING.md` 8번 같은 하루짜리 과도기 오류)가
  분할로 잘못 감지될 가능성이 있음 — `분할이벤트`/`분할조정factor` 컬럼을 출력에 포함해뒀으니
  의심되는 종목은 직접 확인할 것.
- `004730`/`217590`(PREPROCESSING.md 7번, 사실상 결측)은 행 수가 너무 적어서(`MIN_ROWS=30`
  미만) 자동으로 건너뜀.

## 출력 형식

`raw/daily_price_adjusted/{티커}.csv` — `raw/daily_price`의 원본 컬럼(티커, 날짜, 시가, 고가,
저가, 종가, 거래량, 거래대금, 등락률) + 추가 컬럼:

| 컬럼 | 설명 |
|---|---|
| `수정시가`/`수정고가`/`수정저가`/`수정종가` | 분할/무상증자 보정된 가격 |
| `수정등락률` | 수정종가 기준 일별 변화율(%) |
| `분할조정factor` | 그 날짜에 적용된 누적 보정 비율(1.0=보정 없음) |
| `분할이벤트` | 그 날짜가 분할/무상증자 이벤트로 감지됐는지(bool) |

## 실행법

```bash
tmux new -s adjust-price
uv run python daily-price-split-adjust/adjust_daily_price.py
```

분리: `Ctrl+b d` / 재접속: `tmux attach -t adjust-price`

실행 로그(감지된 이벤트 목록 포함)는 `daily-price-split-adjust/log/adjust_daily_price.log`에
남음.
