"""
검토 STEP 1: downloads/ 폴더에 있는 파일명(bigkinds_YYYY-MM-DD.xlsx)을 기준으로, 데이터가
존재하는 연도 범위 안에서 빠진 날짜(파일이 없는 날짜)를 연도별로 찾아 출력한다.
(step2_verify_downloads.py의 날짜 무결성 검사와는 별개로, "파일 자체가 없는 날짜"를 찾는 스크립트)

사용법:
    cd bigkinds-crawler
    uv run python step1_find_missing_dates.py
"""

import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

DOWNLOAD_DIR = Path(__file__).resolve().parent / "downloads"
FILENAME_PATTERN = re.compile(r"bigkinds_(\d{4})-(\d{2})-(\d{2})\.xlsx$")


def collect_existing_dates() -> set[date]:
    existing = set()
    for path in DOWNLOAD_DIR.glob("bigkinds_*.xlsx"):
        m = FILENAME_PATTERN.search(path.name)
        if not m:
            continue
        year, month, day = (int(g) for g in m.groups())
        existing.add(date(year, month, day))
    return existing


def missing_dates_by_year(existing: set[date]) -> dict[int, list[date]]:
    years = sorted({d.year for d in existing})
    missing: dict[int, list[date]] = defaultdict(list)

    for year in years:
        cursor = date(year, 1, 1)
        year_end = date(year, 12, 31)
        while cursor <= year_end:
            if cursor not in existing:
                missing[year].append(cursor)
            cursor += timedelta(days=1)

    return missing


def main() -> None:
    existing = collect_existing_dates()
    if not existing:
        print("downloads/ 에서 날짜 패턴에 맞는 파일을 찾지 못했습니다.")
        return

    years = sorted({d.year for d in existing})
    print(f"데이터 존재 연도 범위: {years[0]} ~ {years[-1]}")
    print()

    missing = missing_dates_by_year(existing)
    total_missing = sum(len(v) for v in missing.values())

    for year in years:
        year_missing = missing.get(year, [])
        if not year_missing:
            print(f"{year}: 누락 없음")
            continue
        print(f"{year}: {len(year_missing)}일 누락")
        for d in year_missing:
            print(f"   {d.isoformat()}")

    print()
    print(f"총 누락 일수: {total_missing}")


if __name__ == "__main__":
    main()
