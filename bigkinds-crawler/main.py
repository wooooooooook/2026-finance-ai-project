"""
BigKinds 뉴스 엑셀 다운로드를 날짜 단위로 진행하며, 진행 상태를
2026-finance-ai-news-server의 /api/crawl-dates API로 관리하는 스크립트.

- 날짜 하나를 claim -> 다운로드 -> complete 순으로 반복한다.
- 서버가 날짜별로 락(TTL)을 걸어주므로, 여러 명이 동시에 이 스크립트를 실행해도
  서로 다른 날짜를 집어가며 겹치지 않는다. 워커가 중간에 죽어도 TTL이 지나면
  다른 워커가 그 날짜를 다시 집어간다.
- 시작할 때 BIGKINDS_USERNAME/PASSWORD로 자동 로그인한 뒤 바로 크롤링을 시작한다.
  (팀원별로 자신의 계정으로 교체해서 사용)
- 다운로드 시작에 실패하면 그 날짜를 서버에 unlock 요청(release)하고 스크립트를 중단한다
  (재시도 없이 종료 - 다시 실행하면 로그인부터 새로 하고 그 날짜부터 재개됨).

사용법:
    cd bigkinds-crawler
    uv pip install -r requirements.txt
    uv run playwright install chromium
    uv run python main.py
"""

import contextlib
import datetime as dt
import itertools
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import requests
from playwright.sync_api import Page, sync_playwright

DOWNLOAD_DIR = Path("downloads")

# 며칠만 쓰고 버릴 스크립트라 하드코딩. 서버 .env.local의 CRAWL_API_KEY와 동일한 값을 넣을 것.
API_BASE_URL = "https://finance-ai-977776325682.us-west1.run.app/"
API_KEY = "a9fbc49dee1765380c2b9166c3d9d80f31f96768298fdc1473dffd4265844f2f"
WORKER_ID = f"{os.environ.get('USER', 'worker')}-{uuid.uuid4().hex[:8]}"

BIGKINDS_ROOT_URL = "https://www.bigkinds.or.kr/"
SEARCH_URL = "https://www.bigkinds.or.kr/v2/news/search.do"

# BigKinds 로그인 계정. 팀원 각자 자신의 계정으로 교체해서 사용.
BIGKINDS_USERNAME = "YOUR_USERNAME"
BIGKINDS_PASSWORD = "YOUR_PASSWORD"

session = requests.Session()
session.headers["Authorization"] = f"Bearer {API_KEY}"

# 다운로드 버튼 클릭 후 다운로드가 시작되기까지 걸린 시간의 누적값
total_wait_seconds = 0.0

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# 브라우저 콘솔/페이지 에러 메시지를 모아두는 버퍼 (실패 시점에 터미널로 출력하기 위함)
console_messages: list[str] = []


# page에 콘솔/JS 에러 메시지를 console_messages에 쌓는 리스너를 연결한다.
def attach_console_logging(page: Page) -> None:
    page.on("console", lambda msg: console_messages.append(f"[console:{msg.type}] {msg.text}"))
    page.on("pageerror", lambda err: console_messages.append(f"[pageerror] {err}"))
    page.on(
        "requestfailed",
        lambda req: console_messages.append(
            f"[requestfailed] {req.method} {req.url} - {req.failure}"
        ),
    )


# 실패 시점에 브라우저 상태(URL, 콘솔 로그, 스크린샷)를 터미널에 출력하고 파일로 남긴다.
def dump_debug_info(page: Page, target_date: dt.date | None) -> None:
    print("--- 디버그 정보 ---")
    try:
        print(f"현재 URL: {page.url}")
        print(f"페이지 제목: {page.title()}")
    except Exception as e:
        print(f"(URL/제목 조회 실패: {type(e).__name__}: {e})")

    if console_messages:
        print(f"브라우저 콘솔/네트워크 로그 ({len(console_messages)}개):")
        for line in console_messages[-30:]:
            print(f"  {line}")
    else:
        print("브라우저 콘솔 로그 없음")

    try:
        DOWNLOAD_DIR.mkdir(exist_ok=True)
        suffix = target_date.isoformat() if target_date else "unknown"
        screenshot_path = DOWNLOAD_DIR / f"error_{suffix}_{int(time.time())}.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"스크린샷 저장: {screenshot_path}")
    except Exception as e:
        print(f"(스크린샷 저장 실패: {type(e).__name__}: {e})")
    print("-------------------")


# 오래 걸리는 동기 작업(Playwright 대기 등) 중 터미널에 회전 스피너를 표시한다.
@contextlib.contextmanager
def spinner(message: str):
    stop = threading.Event()

    def spin() -> None:
        for frame in itertools.cycle(_SPINNER_FRAMES):
            if stop.is_set():
                break
            print(f"\r{frame} {message}", end="", flush=True)
            time.sleep(0.06)
        print(f"\r{' ' * (len(message) + 2)}\r", end="", flush=True)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


# 서버에서 아직 처리되지 않은(또는 락이 만료된) 날짜 하나를 할당받는다.
def claim_date() -> dt.date | None:
    resp = session.post(
        f"{API_BASE_URL}/api/crawl-dates/claim", json={"workerId": WORKER_ID}, timeout=10
    )
    resp.raise_for_status()
    date_str = resp.json().get("date")
    return dt.date.fromisoformat(date_str) if date_str else None


# 해당 날짜를 완료 상태로 서버에 기록한다.
def complete_date(target_date: dt.date) -> None:
    resp = session.post(
        f"{API_BASE_URL}/api/crawl-dates/{target_date.isoformat()}/complete",
        json={"workerId": WORKER_ID},
        timeout=10,
    )
    resp.raise_for_status()


# 해당 날짜를 다시 미처리(pending) 상태로 되돌린다.
def release_date(target_date: dt.date) -> None:
    resp = session.post(
        f"{API_BASE_URL}/api/crawl-dates/{target_date.isoformat()}/release",
        json={"workerId": WORKER_ID},
        timeout=10,
    )
    resp.raise_for_status()


# STEP 01 패널을 열고 '경제일간지' 카테고리를 선택한다 (최초 1회만 호출).
def select_category(page: Page) -> None:
    page.get_by_role("button", name="STEP 01 뉴스 검색 -").click()
    page.locator("a").filter(has_text="경제일간지").click()
    page.wait_for_timeout(1000)
    # 여기서 STEP01 패널이 열린 채로 남아있으면, set_date_range()의 첫 STEP01 클릭이
    # "열기"가 아니라 "닫기"로 토글되어 버린다(이미 열려있어서). 그러면 그 안의 "기간"
    # 버튼이 사라져서 영원히 못 찾고 타임아웃 난다. 항상 "닫힌 상태"로 맞춰두기 위해
    # STEP01을 한 번 더 눌러 닫는다.
    page.get_by_role("button", name="STEP 01 뉴스 검색 -").click()


# 캘린더를 클릭하지 않고 날짜 입력창의 value를 직접 주입한다.
def set_input_date(page: Page, selector: str, target_date: dt.date) -> None:
    date_str = target_date.strftime("%Y-%m-%d")
    page.eval_on_selector(
        selector,
        """(el, value) => {
            el.value = value;
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }""",
        date_str,
    )


# 최초 호출인지 추적한다 (입력창 기본값이 오늘 날짜라 최초 호출만 순서가 반대로 안전함).
_date_range_initialized = False


# 검색 시작일/종료일을 같은 날짜로 설정하고 적용한다.
def set_date_range(page: Page, target_date: dt.date) -> None:
    global _date_range_initialized

    # STEP 01 패널로 돌아와서 기간 설정 하위 패널을 연다.
    page.get_by_role("button", name="STEP 01 뉴스 검색 -").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="기간").click()

    # 시작일 <= 종료일 제약을 잠깐이라도 어기지 않도록 순서를 정한다.
    # - 최초 호출: 입력창 기본값이 오늘 날짜라 target_date보다 미래이므로 시작일부터 당긴다.
    # - 이후 호출: 날짜가 오름차순으로 진행되므로 종료일부터 늘린다.
    if _date_range_initialized:
        set_input_date(page, "#search-end-date", target_date)
        set_input_date(page, "#search-begin-date", target_date)
    else:
        set_input_date(page, "#search-begin-date", target_date)
        set_input_date(page, "#search-end-date", target_date)
        _date_range_initialized = True

    page.get_by_role("button", name="적용하기").click()


# 루트 페이지에서 로그인한 뒤 검색 화면 URL로 직접 이동한다.
def login(page: Page) -> None:
    with spinner("로그인 시도 중"):
        page.goto(BIGKINDS_ROOT_URL)
        page.get_by_role("link", name="MEMBERSHIP").click()
        page.get_by_role("link", name="로그인").click()
        page.get_by_role("textbox", name="이메일주소 입력").fill(BIGKINDS_USERNAME)
        page.get_by_role("textbox", name="비밀번호 입력").fill(BIGKINDS_PASSWORD)
        page.get_by_role("textbox", name="비밀번호 입력").press("Enter")
        page.wait_for_load_state("domcontentloaded")
        # 로그인 처리(세션/쿠키 설정)가 domcontentloaded보다 늦게 끝나는 경우가 있어서,
        # 로그인이 실제로 끝나기 전에 다음 단계로 넘어가지 않도록 추가로 기다린다.
        page.wait_for_timeout(2000)
        # "검색" 링크 클릭이 가끔 멈춰서, 로그인 후 검색 화면으로 바로 이동한다.
        # 로그인 후 페이지 쪽에서 자체적으로 또 리다이렉트를 하는 경우가 있어서
        # 우리가 보낸 goto가 net::ERR_ABORTED로 취소될 수 있다. 그 경우 잠깐
        # 기다렸다가 재시도한다.
        for attempt in range(5):
            try:
                page.goto(SEARCH_URL)
                break
            except Exception:
                if attempt == 4:
                    raise
                page.wait_for_timeout(500)
        # 검색 페이지 로드 중 발생하는 모든 네트워크 요청이 끝날 때까지 기다린 뒤 다음 단계로 넘어간다.
        page.wait_for_load_state("networkidle")
    print("로그인 완료")


# STEP 03에서 엑셀 다운로드를 시도한다. 실패하면 예외를 그대로 올린다 (재시도 없음).
def download_excel(page: Page, target_date: dt.date) -> Path:
    global total_wait_seconds

    wait_start = time.perf_counter()
    page.get_by_role("button", name="STEP 03 분석 결과 및 시각화").click()
    # networkidle은 광고/분석 등 백그라운드 요청이 계속 있으면 절대 idle이 되지 않아
    # 매번 타임아웃이 날 수 있다. 실제로 필요한 조건(엑셀 다운로드 버튼이 보이는 것)을
    # 직접 기다리는 쪽이 더 안정적이다.
    page.get_by_role("button", name="엑셀 다운로드").wait_for(state="visible", timeout=60_000)

    with spinner("엑셀 다운로드 버튼 클릭 완료 - 다운로드 대기 중"):
        with page.expect_download(timeout=60_000) as download_info:
            page.get_by_role("button", name="엑셀 다운로드").click()
    download = download_info.value

    elapsed = time.perf_counter() - wait_start
    total_wait_seconds += elapsed
    print(f"다운로드 시작됨 (대기: {elapsed:.1f}초, 누적 대기시간: {total_wait_seconds:.1f}초)")

    DOWNLOAD_DIR.mkdir(exist_ok=True)
    dest = DOWNLOAD_DIR / f"bigkinds_{target_date.isoformat()}.xlsx"
    download.save_as(dest)
    return dest


# 로그인 후 날짜를 하나씩 claim하여 다운로드/완료 처리하는 메인 루프.
def run() -> None:
    print(f"워커 ID: {WORKER_ID}")

    with sync_playwright() as p:
        # GPU 프로세스가 EXC_BREAKPOINT(SIGTRAP)로 크래시하는 문제(크래시 리포트로 확인됨)가 있어
        # GPU 가속을 꺼서 실행한다.
        browser = p.chromium.launch(headless=False, args=["--disable-gpu"])
        context = browser.new_context()
        page = context.new_page()
        # 엑셀 다운로드 클릭 시 뜨는 confirm() 다이얼로그를 자동으로 "확인" 처리한다.
        # (Playwright는 핸들러가 없으면 다이얼로그를 기본 dismiss 하므로 다운로드가 시작되지 않는다)
        page.on("dialog", lambda dialog: dialog.accept())
        page.set_default_timeout(60_000)  # 모든 동작(클릭/대기)의 기본 타임아웃을 30초 -> 60초로
        attach_console_logging(page)

        login(page)

        select_category(page)

        target_date = None
        try:
            while True:
                target_date = claim_date()
                if target_date is None:
                    print("더 이상 처리할 날짜가 없습니다.")
                    break

                with spinner(f"처리 중: {target_date}"):
                    set_date_range(page, target_date)
                dest = download_excel(page, target_date)
                complete_date(target_date)
                print(f"[OK] {target_date} -> {dest}")
        except KeyboardInterrupt:
            if target_date is not None:
                print(f"중단됨. {target_date}를 다시 처리 대기 상태로 되돌립니다.")
                release_date(target_date)
        except requests.exceptions.ConnectionError:
            print(f"서버({API_BASE_URL})에 연결할 수 없습니다. 서버가 켜져 있는지 확인하세요 (예: pnpm dev).")
            if target_date is not None:
                print(f"{target_date}는 unlock 요청도 보낼 수 없어, 락 TTL이 지나야 다른 워커가 다시 집어갑니다.")
            context.close()
            browser.close()
            sys.exit(1)
        except Exception as e:
            print(f"[중단] {target_date} 처리 중 오류 발생: {type(e).__name__}: {e}")
            dump_debug_info(page, target_date)
            if target_date is not None:
                try:
                    release_date(target_date)
                    print(f"{target_date}를 다시 처리 대기 상태로 되돌렸습니다.")
                except Exception as release_error:
                    print(
                        f"{target_date} unlock 요청 실패: {type(release_error).__name__}: {release_error} "
                        "(락 TTL이 지나면 다른 워커가 다시 집어갑니다)"
                    )
            context.close()
            browser.close()
            sys.exit(1)

        context.close()
        browser.close()

    print("처리 종료")


if __name__ == "__main__":
    run()
