"""
downloads/ 폴더의 모든 엑셀 파일에 대해, 파일명 날짜와 "일자" 컬럼 값이
전부 일치하는지 검사한다 (날짜 무결성 검사). 누락된 날짜는 계산하지 않는다.

사용법:
    cd bigkinds-crawler
    uv run python verify_downloads.py
"""

import re
from collections import Counter
from pathlib import Path

import openpyxl

# 실행 위치(cwd)와 상관없이 항상 이 스크립트 옆의 downloads/ 폴더를 보도록, 경로를
# 현재 작업 디렉터리가 아니라 이 파일 위치 기준으로 잡는다.
DOWNLOAD_DIR = Path(__file__).resolve().parent / "downloads"
FILENAME_PATTERN = re.compile(r"bigkinds_(\d{4})-(\d{2})-(\d{2})\.xlsx$")


# 파일명에서 기대하는 날짜(YYYYMMDD)를 추출한다.
def expected_date(path: Path) -> str | None:
    m = FILENAME_PATTERN.search(path.name)
    if not m:
        return None
    return m.group(1) + m.group(2) + m.group(3)


# 엑셀의 "일자" 컬럼(두 번째 컬럼)에 실제로 어떤 값들이 몇 번씩 들어있는지 센다.
def actual_dates(path: Path) -> Counter:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    dates: Counter = Counter()
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
        if row[0] is None and all(cell is None for cell in row):
            continue
        dates[row[1]] += 1
    return dates


# 터미널에 진행 상황 막대그래프를 찍는다 (같은 줄을 덮어써서 갱신).
def print_progress(current: int, total: int, width: int = 40) -> None:
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    percent = (current / total * 100) if total else 100.0
    print(f"\r[{bar}] {current}/{total} ({percent:5.1f}%)", end="", flush=True)


def main() -> None:
    files = sorted(DOWNLOAD_DIR.glob("bigkinds_*.xlsx"))
    print(f"총 {len(files)}개 파일 검사")

    problem_files: list[Path] = []
    for i, path in enumerate(files, start=1):
        expected = expected_date(path)
        if expected is None:
            print(f"\nSKIP (파일명 패턴 불일치): {path.name}")
            print_progress(i, len(files))
            continue

        try:
            dates = actual_dates(path)
        except Exception as e:
            print(f"\nLOAD_FAIL {path.name}: {type(e).__name__}: {e}")
            problem_files.append(path)
            print_progress(i, len(files))
            continue

        total = sum(dates.values())
        ok = total > 0 and dict(dates) == {expected: total}
        if not ok:
            problem_files.append(path)

        print_progress(i, len(files))

    print()
    if problem_files:
        print(f"문제 있는 파일: {len(problem_files)}개")
        for path in problem_files:
            print(path.name)
    else:
        print("문제 있는 파일 없음")


if __name__ == "__main__":
    main()
