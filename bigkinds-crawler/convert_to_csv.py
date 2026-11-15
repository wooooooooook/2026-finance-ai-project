"""
downloads/의 모든 xlsx 파일을 gzip 압축 csv(.csv.gz)로 그대로 변환해 compressed/에 저장한다.
필터링/전처리는 하지 않는다 - 원본 xlsx의 모든 컬럼/행을 1:1로 csv.gz 포맷에 옮기기만 한다.
xlsx는 그대로 두고 삭제하지 않는다.

파일마다 독립적인 작업이라 멀티프로세스로 병렬 처리한다 (NUM_WORKERS로 조절).

사용법:
    cd bigkinds-crawler
    uv run python convert_to_csv.py
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

NUM_WORKERS = os.cpu_count() or 4

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "compressed"


# 파일 하나를 csv.gz로 변환한다. 별도 프로세스에서 실행된다.
def convert_file(path: Path) -> tuple[Path, str | None]:
    output_path = OUTPUT_DIR / f"{path.stem}.csv.gz"
    if output_path.exists():
        return path, None

    try:
        df = pd.read_excel(path)
        df.to_csv(output_path, index=False, compression="gzip")
        return path, None
    except Exception as e:
        return path, f"{type(e).__name__}: {e}"


def print_progress(current: int, total: int, width: int = 40) -> None:
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    percent = (current / total * 100) if total else 100.0
    print(f"\r[{bar}] {current}/{total} ({percent:5.1f}%)", end="", flush=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DOWNLOAD_DIR.glob("bigkinds_*.xlsx"))
    print(f"총 {len(files)}개 파일 변환 ({NUM_WORKERS}개 프로세스) -> {OUTPUT_DIR}")

    failed: list[tuple[Path, str]] = []
    done = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(convert_file, path): path for path in files}
        for future in as_completed(futures):
            path, error = future.result()
            done += 1
            if error is not None:
                print(f"\nFAIL {path.name}: {error}")
                failed.append((path, error))
            print_progress(done, len(files))

    print()
    if failed:
        print(f"실패: {len(failed)}개")
        for path, error in failed:
            print(f"  {path.name}: {error}")
    else:
        print("전부 성공")


if __name__ == "__main__":
    main()
