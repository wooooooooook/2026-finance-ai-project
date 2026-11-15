"""
검토 STEP 3: step1_find_missing_dates.py(STEP 1, 누락 날짜 탐색), step2_verify_downloads.py
(STEP 2, 날짜 무결성 검사)에 이어서, downloads/의 모든 뉴스 행에 대해 '제목' 또는 '본문'이
결측(빈 값)인 행이 있는지 파악한다.

파일마다 독립적인 작업이라 멀티프로세스로 병렬 처리한다 (NUM_WORKERS로 조절).

사용법:
    cd bigkinds-crawler
    uv run python step3_verify_missing_content.py
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import openpyxl

NUM_WORKERS = os.cpu_count() or 4

DOWNLOAD_DIR = Path(__file__).resolve().parent / "downloads"


def is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


@dataclass
class FileResult:
    name: str
    total_rows: int
    title_missing: int
    body_missing: int
    both_missing: int


# 파일 하나를 검사해서 결측 통계를 돌려준다. 별도 프로세스에서 실행된다.
def check_file(path: Path) -> FileResult:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]

    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    title_idx = header.index("제목")
    body_idx = header.index("본문")

    total = title_missing = body_missing = both_missing = 0
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
        if row[0] is None and all(is_blank(c) for c in row):
            continue  # 완전히 빈 행은 집계 대상에서 제외

        total += 1
        t_missing = is_blank(row[title_idx])
        b_missing = is_blank(row[body_idx])
        title_missing += t_missing
        body_missing += b_missing
        both_missing += t_missing and b_missing

    return FileResult(path.name, total, title_missing, body_missing, both_missing)


def print_progress(current: int, total: int, width: int = 40) -> None:
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    percent = (current / total * 100) if total else 100.0
    print(f"\r[{bar}] {current}/{total} ({percent:5.1f}%)", end="", flush=True)


def main() -> None:
    files = sorted(DOWNLOAD_DIR.glob("bigkinds_*.xlsx"))
    print(f"총 {len(files)}개 파일 검사 ({NUM_WORKERS}개 프로세스)")

    results: list[FileResult] = []
    done = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(check_file, path): path for path in files}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                path = futures[future]
                print(f"\nLOAD_FAIL {path.name}: {type(e).__name__}: {e}")
            done += 1
            print_progress(done, len(files))

    print()

    affected = [r for r in results if r.title_missing or r.body_missing]
    affected.sort(key=lambda r: r.name)

    total_rows = sum(r.total_rows for r in results)
    total_title_missing = sum(r.title_missing for r in results)
    total_body_missing = sum(r.body_missing for r in results)
    total_both_missing = sum(r.both_missing for r in results)

    if affected:
        print(f"제목/본문 결측이 있는 파일: {len(affected)}개")
        for r in affected:
            print(
                f"  {r.name}: 전체 {r.total_rows}행 중 "
                f"제목 결측 {r.title_missing}개, 본문 결측 {r.body_missing}개, "
                f"둘 다 결측 {r.both_missing}개"
            )
    else:
        print("제목/본문 결측 행 없음")

    print()
    print(
        f"총계: 전체 {total_rows}행 중 제목 결측 {total_title_missing}개, "
        f"본문 결측 {total_body_missing}개, 둘 다 결측 {total_both_missing}개"
    )


if __name__ == "__main__":
    main()
