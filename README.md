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
2. `Cmd+Shift+P` (Windows: `Ctrl+Shift+P`) → **"Notebook: Select Notebook Kernel"**
3. 로컬 개발 환경 선택 시 `./.venv/bin/python` workspace의 인터프리터 선택
4. Colab 환경 선택 시 Colab GPU T4 latest 커널 생성 후 사용

### Colab에서 실행하는 경우

Colab 호스팅 런타임은 Python 3.12.13을 기본 제공하므로 별도 Python 설치가 필요 없음. 단, Colab 원격 런타임은 실행 시 초기화 되므로 매번 라이브러리 설치 필요 (template.ipynb 참조)

## Convention

- 파일명은 케밥-케이스 사용
- 새로운 노트북 작업은 `template.ipynb`를 복사해서 사용

## Contributors

[![Contributors](https://contrib.rocks/image?repo=wooooooooook/2026-finance-ai-project)](https://github.com/wooooooooook/2026-finance-ai-project/graphs/contributors)
