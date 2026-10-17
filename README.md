# 2026 금융데이터인텔리전스

코스피(KOSPI) 상장·과거상장 보통주를 대상으로, 월말까지 이용 가능했던 정보로 다음 달 개별 종목 수익률을 예측하는 프로젝트입니다.

## Prerequisite

- [uv](https://github.com/astral-sh/uv)

## Installation (최초 실행 시 셋업)

### 1. uv 설치

[uv](https://github.com/astral-sh/uv#installation)

설치 후 터미널 재시작

### 2. Python 3.12.13 설치

```bash
uv python install 3.12.13
```

### 3. 가상환경 생성

```bash
uv venv --python 3.12.13
```

프로젝트 루트에 `.venv` 폴더가 생성됩니다

### 4. 의존성 패키지 설치

```bash
uv pip install -r requirements.txt
```

## Starts (매번 작업 전 실행)

### VSCode 인터프리터 선택

1. VSCode에서 이 프로젝트 폴더를 열고, Python 확장이 설치되어 있는지 확인
2. `Cmd+Shift+P` (Windows: `Ctrl+Shift+P`) → **"Python: Select Interpreter"**
3. `./.venv/bin/python` workspace의 인터프리터 선택

## Google Colab에서 개발하는 경우

Colab 호스팅 런타임은 Python 3.12.13을 기본 제공하므로 별도 Python 설치가 필요 없음. 대신 노트북 첫 셀에 아래 명령어를 넣고 의존성 패키지를 작업 전 매번 설치.

```python
!pip install -r requirements.txt -q
```

## 제작 지원

[![Contributors](https://contrib.rocks/image?repo=wooooooooook/2026-finance-ai-project)](https://github.com/wooooooooook/2026-finance-ai-project/graphs/contributors)
