"""
step1_find_missing_dates.py가 찾아낸 "파일 자체가 없는 날짜"를 여러 프로세스로 나눠 병렬
크롤링하는 스크립트. 각 프로세스가 독립적으로 로그인해서 별도 브라우저를 띄운다
(BigKinds가 동일 계정 동시 세션을 허용하는 것을 확인함).

Playwright의 동기(sync) API는 스레드 간 전환이 안 되는 제약(greenlet 기반)이 있어서,
스레드가 아니라 완전히 별도의 OS 프로세스로 띄운다 - 프로세스마다 독립적인 Playwright
드라이버를 가지므로 서로 간섭하지 않는다.

날짜 하나가 실패해도 그 프로세스는 중단하지 않고 다음 날짜로 넘어가며, 전체가 끝나면
성공/실패 목록을 출력한다.

사용법:
    cd bigkinds-crawler
    uv run python crawl_missing_dates.py
"""

import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed

from playwright.sync_api import Page, sync_playwright

import main
from step1_find_missing_dates import collect_existing_dates, missing_dates_by_year

NUM_WORKERS = 4


def get_missing_dates() -> list[dt.date]:
    existing = collect_existing_dates()
    missing_by_year = missing_dates_by_year(existing)
    return sorted(d for year_dates in missing_by_year.values() for d in year_dates)


def chunk(dates: list[dt.date], n: int) -> list[list[dt.date]]:
    shards: list[list[dt.date]] = [[] for _ in range(n)]
    for i, d in enumerate(dates):
        shards[i % n].append(d)
    return shards


# main.set_date_range()와 동일하지만, "첫 호출 여부"를 전역이 아니라 이 워커 전용 state에 둔다.
# (이 프로세스는 자기 혼자만 페이지를 다루므로 사실 전역 변수를 써도 되지만, main.py를
# 그대로 재사용하면서 혼동을 피하기 위해 독립적으로 둔다)
def set_date_range_isolated(page: Page, target_date: dt.date, state: dict) -> None:
    page.get_by_role("button", name="STEP 01 뉴스 검색 -").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="기간").click()

    if state["initialized"]:
        main.set_input_date(page, "#search-end-date", target_date)
        main.set_input_date(page, "#search-begin-date", target_date)
    else:
        main.set_input_date(page, "#search-begin-date", target_date)
        main.set_input_date(page, "#search-end-date", target_date)
        state["initialized"] = True

    page.get_by_role("button", name="적용하기").click()


# 별도 프로세스에서 실행된다: 독립적으로 로그인부터 시작해서 자기 몫의 날짜를 전부 처리한다.
def worker(worker_id: int, dates: list[dt.date]) -> tuple[list[dt.date], list[dt.date]]:
    succeeded: list[dt.date] = []
    failed: list[dt.date] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--disable-gpu"])
        context = browser.new_context()
        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.accept())
        page.set_default_timeout(60_000)
        main.attach_console_logging(page)

        main.login(page)
        main.select_category(page)

        date_state = {"initialized": False}

        for i, target_date in enumerate(dates, start=1):
            print(f"[worker {worker_id}][{i}/{len(dates)}] {target_date} 처리 중...")
            try:
                with main.spinner(f"[worker {worker_id}] 처리 중: {target_date}"):
                    set_date_range_isolated(page, target_date, date_state)
                dest = main.download_excel(page, target_date)
                print(f"[worker {worker_id}][OK] {target_date} -> {dest}")
                succeeded.append(target_date)
            except Exception as e:
                print(f"[worker {worker_id}][FAIL] {target_date}: {type(e).__name__}: {e}")
                main.dump_debug_info(page, target_date)
                failed.append(target_date)

        context.close()
        browser.close()

    return succeeded, failed


def run(num_workers: int) -> None:
    missing = get_missing_dates()
    print(f"누락된 날짜 {len(missing)}개 발견, {num_workers}개 프로세스로 분담")
    if not missing:
        return

    shards = chunk(missing, num_workers)
    for i, shard in enumerate(shards):
        print(f"  worker {i}: {len(shard)}개")

    all_succeeded: list[dt.date] = []
    all_failed: list[dt.date] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(worker, i, shard): i
            for i, shard in enumerate(shards)
            if shard
        }
        for future in as_completed(futures):
            succeeded, failed = future.result()
            all_succeeded.extend(succeeded)
            all_failed.extend(failed)

    print()
    print(f"완료: 성공 {len(all_succeeded)}개 / 실패 {len(all_failed)}개")
    if all_failed:
        print("실패한 날짜:")
        for d in sorted(all_failed):
            print(f"   {d.isoformat()}")


if __name__ == "__main__":
    run(NUM_WORKERS)
