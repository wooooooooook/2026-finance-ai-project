"""
검토 STEP 2: downloads/ 폴더의 모든 엑셀 파일에 대해, 파일명 날짜와 "일자" 컬럼 값이
전부 일치하는지 검사한다 (날짜 무결성 검사). step1_find_missing_dates.py와 달리
파일이 없는 날짜는 다루지 않고, 존재하는 파일 내부의 날짜 일관성만 검사한다.

파일마다 독립적인 작업이라 멀티프로세스로 병렬 처리한다 (NUM_WORKERS로 조절).

사용법:
    cd bigkinds-crawler
    uv run python step2_verify_downloads.py
"""

import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import openpyxl

NUM_WORKERS = os.cpu_count() or 4

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
# 주의: 이 파일들은 read_only=True 모드에서 max_row/max_column이 (1, 1)로 잘못 인식되어
# (dimension 메타데이터 문제로 보임) 행을 하나도 못 읽는 버그가 있어 read_only는 쓰지 않는다.
def actual_dates(path: Path) -> Counter:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    dates: Counter = Counter()
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
        if row[0] is None and all(cell is None for cell in row):
            continue
        dates[row[1]] += 1
    return dates


# 파일 하나를 검사해서 상태를 돌려준다. 별도 프로세스에서 실행되므로 전역 상태에 의존하지 않는다.
def check_file(path: Path) -> tuple[Path, str | None]:
    expected = expected_date(path)
    if expected is None:
        return path, "SKIP (파일명 패턴 불일치)"

    try:
        dates = actual_dates(path)
    except Exception as e:
        return path, f"LOAD_FAIL: {type(e).__name__}: {e}"

    total = sum(dates.values())
    ok = total > 0 and dict(dates) == {expected: total}
    return path, None if ok else "MISMATCH"


# 터미널에 진행 상황 막대그래프를 찍는다 (같은 줄을 덮어써서 갱신).
def print_progress(current: int, total: int, width: int = 40) -> None:
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    percent = (current / total * 100) if total else 100.0
    print(f"\r[{bar}] {current}/{total} ({percent:5.1f}%)", end="", flush=True)


def main() -> None:
    files = sorted(DOWNLOAD_DIR.glob("bigkinds_*.xlsx"))
    print(f"총 {len(files)}개 파일 검사 ({NUM_WORKERS}개 프로세스)")

    problem_files: list[tuple[Path, str]] = []
    done = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(check_file, path): path for path in files}
        for future in as_completed(futures):
            path, problem = future.result()
            done += 1
            if problem is not None:
                print(f"\n{problem}: {path.name}")
                if not problem.startswith("SKIP"):
                    problem_files.append((path, problem))
            print_progress(done, len(files))

    print()
    if problem_files:
        problem_files.sort(key=lambda item: item[0].name)
        print(f"문제 있는 파일: {len(problem_files)}개")
        for path, problem in problem_files:
            print(f"{path.name}: {problem}")
    else:
        print("문제 있는 파일 없음")


if __name__ == "__main__":
    main()
