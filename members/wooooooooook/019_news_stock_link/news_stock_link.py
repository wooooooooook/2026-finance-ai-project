"""
news_stock_link.py: 019 실험 1단계 — BigKinds 경제일간지 기사를 종목에 연결하고, 종목·월별 기사 수와
커버리지(어떤 종목·연도에 기사가 얼마나 연결되는지)를 확인한다. 수익률은 아직 쓰지 않는다.

v2(정밀도 개선): v1에서 일반명사형 2글자 이름("대상", "남성" 등)과 그룹명 접두("LG텔레콤" → lg), 증권사
바이라인, 시황·공시 나열 기사의 오탐이 많았다. 규칙은 README의 "매칭 규칙" 참고. v1 결과는 results/v1_initial/.

실행 순서:
  1. link:   시점별 종목명 사전 → 기사 제목·기관 태그·본문에서 종목명 매칭 → 종목×월 기사 수 집계
  2. report: 커버리지 표·플롯, 매칭 정확도 점검용 표본(직접 라벨링하면 정밀도 계산)

2001-01 ~ 2021-12만 사용한다(test 2022~2025는 변수·방법 선택에 쓰지 않는다는 규칙).

사용법:
    uv run python members/wooooooooook/019_news_stock_link/news_stock_link.py          # link + report
    uv run python members/wooooooooook/019_news_stock_link/news_stock_link.py report   # 집계 결과로 report만
    uv run python members/wooooooooook/019_news_stock_link/news_stock_link.py v1plots  # v1 결과(results/v1_initial)로 도표만
    uv run python members/wooooooooook/019_news_stock_link/news_stock_link.py variantplots  # v1 / v2 / 기관만 매칭률 도표만
"""

import os
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from tqdm import tqdm

# ===========================================================================
# 설정
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
RAW = PROJECT_ROOT / "raw"
NEWS_DIR = RAW / "news"

FEATURES_DIR = BASE_DIR / "features"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"
COUNTS_PATH = FEATURES_DIR / "stock_month_news_counts.csv.gz"
LABELED_SAMPLES = RESULTS_DIR / "match_samples_labeled.csv"

MONTH_START, MONTH_END = "2001-01", "2021-12"
EXCLUDED_CODES = {"004730", "217590"}
MONTH_TOLERANCE = 1       # 종목명이 바뀌기 전후 ±N개월은 같은 종목으로 인정
N_WORKERS = min(os.cpu_count() or 1, 48)

# --- 매칭 규칙(v2) ---
MIN_NAME_LEN = 3          # 정규화 후 이 길이 미만의 종목명은 제외(일반명사·그룹명 오탐)
ALLOW_SHORT_NAMES = {"kt"}  # 예외로 허용하는 짧은 이름(뒤에 영숫자가 붙으면 다른 이름이라 제외됨)
MAX_CODES_PER_ARTICLE = 4   # 제목·기관 태그에서 이 개수 이상 종목이 잡히면 시황·공시 나열 기사로 보고 제외
MARKET_REPORT_KEYWORDS = ("시황", "매매동향", "동시호가", "거래소공시", "시세표")
# 종목명 바로 뒤에 올 수 있는 조사(그 외 한글이 바로 이어지면 다른 단어의 일부로 보고 제외: "LG텔레콤" ≠ lg)
PARTICLES = ("에서", "에게", "으로", "까지", "부터", "보다", "처럼", "은", "는", "이", "가", "을", "를",
             "의", "에", "도", "와", "과", "로", "만", "께", "측", "발")
# 제목 끝 "-증권사" 바이라인(예: "...상향-sk", "- 메리츠")은 종목이 아니라 출처라서 제거
BROKERS = ["신한", "메리츠", "하나", "대신", "삼성", "한화", "현대", "교보", "동양", "대우", "lg", "sk", "굿모닝",
           "동부", "우리", "nh", "kb", "유진", "이트레이드", "키움", "한투", "미래에셋", "ibk", "하이", "부국",
           "신영", "cj", "한양", "동원", "서울", "bnk", "dgb", "푸르덴셜", "리딩", "유화", "한국"]
BROKER_TAIL = re.compile(
    r"\s*[-–—]\s*(?:" + "|".join(map(re.escape, BROKERS)) + r")(?:증권|투자증권|투자|금융투자|종금|리서치)?\s*$")
ASCII_CONT = re.compile(r"\s[a-z0-9]")  # 이름이 영문으로 끝나고 뒤에 영문 단어가 이어지면("lg cns") 다른 회사

SAMPLES_PER_FILE = 3
SAMPLES_PER_STRATUM = 50  # (매칭 유형 × 이름 길이) 층별 최종 표본 수

SOURCES = ["title", "org", "body", "core"]   # core = 제목 ∪ 기관 태그 (본문만 걸린 기사는 제외)
# org_only = 제목 매칭 없이 기관 태그로만 걸린 기사 수(기사 단위, 같은 달 다른 기사가 제목에서 걸려도 셈)
COUNT_COLS = SOURCES + ["org_only"]
SOURCE_LABEL_KO = {"title": "제목", "org": "기관 태그", "body": "본문(200자)", "core": "제목+기관 태그"}
SOURCE_LABEL_EN = {"title": "title", "org": "org tag", "body": "body (200 chars)", "core": "title + org tag"}
SIZE_LABELS = ["small", "mid", "large"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
SIZE_COLORS = {"small": "#2a78d6", "mid": "#eb6834", "large": "#1baf7a"}


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ===========================================================================
# 종목명 사전 (시점별)
# ===========================================================================

def norm_text(s: str) -> str:
    """종목명·기사 텍스트 공통 정규화. KRX 종목명은 '&'가 빠져 있어(KT&G → KTG) 양쪽에서 같이 지운다."""
    return s.replace("&", "").replace("㈜", "").replace("(주)", "").lower()


def build_alias() -> tuple[dict, pd.DataFrame]:
    """alias[정규화된 종목명][월] = 종목코드 (그 달 그 이름을 쓴 종목이 하나일 때만. 둘 이상이면 모호해서 제외)."""
    names = pd.read_csv(RAW / "returns" / "train_2001_2021__company_name.csv", dtype=str, keep_default_na=False)
    names = names.set_index("month")
    names = names[(names.index >= MONTH_START) & (names.index <= MONTH_END)]
    months = list(names.index)

    by_name: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for code in names.columns:
        if code in EXCLUDED_CODES:
            continue
        for i, nm in enumerate(names[code].tolist()):
            n = norm_text(nm.strip())
            if len(n) < MIN_NAME_LEN and n not in ALLOW_SHORT_NAMES:
                continue
            for j in range(i - MONTH_TOLERANCE, i + MONTH_TOLERANCE + 1):
                if 0 <= j < len(months):
                    by_name[n][months[j]].add(code)

    alias, rows = {}, []
    for n, by_month in by_name.items():
        alias[n] = {m: next(iter(cs)) for m, cs in by_month.items() if len(cs) == 1}
        codes = sorted({c for cs in by_month.values() for c in cs})
        rows.append({
            "name": n, "name_len": len(n), "n_codes": len(codes), "codes": ",".join(codes),
            "first_month": min(by_month), "last_month": max(by_month),
            "n_months": len(by_month), "n_ambiguous_months": sum(len(cs) > 1 for cs in by_month.values()),
        })
    return alias, pd.DataFrame(rows).sort_values(["name_len", "name"])


def _trie_regex(node: dict) -> str:
    """종목명 목록을 트라이로 접어 하나의 정규식으로 만든다(가장 긴 이름이 우선 매칭)."""
    branches = [re.escape(ch) + _trie_regex(node[ch]) for ch in sorted(k for k in node if k != "")]
    if not branches:
        return ""
    body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
    return f"(?:{body})?" if "" in node else body


def build_pattern(names) -> str:
    trie: dict = {}
    for name in names:
        node = trie
        for ch in name:
            node = node.setdefault(ch, {})
        node[""] = {}
    # 앞: 이름 앞이 한글·영숫자면 단어 중간이므로 제외
    # 뒤: 조사(선택) 다음이 한글·영숫자가 아니어야 함 → "LG텔레콤", "삼성전자서비스" 같은 복합어 제외.
    #     가장 긴 이름이 뒤 조건을 못 지키면 정규식이 더 짧은 이름으로 되돌아가 다시 시도한다
    suffix = r"(?=(?:" + "|".join(PARTICLES) + r")?(?![0-9A-Za-z가-힣]))"
    return r"(?<![0-9A-Za-z가-힣])" + _trie_regex(trie) + suffix


# ===========================================================================
# 단계 1: 기사 ↔ 종목 연결 (파일 단위 병렬)
# ===========================================================================

_PATTERN = None
_ALIAS: dict = {}


def _init_worker(pattern: str, alias: dict) -> None:
    global _PATTERN, _ALIAS
    _PATTERN = re.compile(pattern)
    _ALIAS = alias


def find_names(text: str, month_alias: dict) -> set:
    found = set()
    for m in _PATTERN.finditer(text):
        name = m.group(0)
        if name not in month_alias:
            continue
        if name[-1].isascii() and ASCII_CONT.match(text, m.end()):
            continue
        found.add(name)
    return found


def process_file(path_str: str):
    path = Path(path_str)
    month = path.stem.replace("bigkinds_", "")[:7]
    d = pd.read_csv(path, usecols=["제목", "본문", "기관", "분석제외 여부"], dtype=str,
                    keep_default_na=False, encoding="utf-8-sig")
    # PREPROCESSING.md 1~4·6번: 빈 제목/본문, 분석제외, 제목+본문 중복, 반복 제목 제거
    d = d[(d["제목"].str.strip() != "") & (d["본문"].str.strip() != "")]
    d = d[d["분석제외 여부"].str.strip() == ""]
    d = d.drop_duplicates(["제목", "본문"])
    vc = d["제목"].value_counts()
    bad = vc[(vc >= 10) & (vc >= 0.01 * len(d))].index
    d = d[~d["제목"].isin(bad)]

    month_alias = {n: by_m[month] for n, by_m in _ALIAS.items() if month in by_m}
    counts: dict = defaultdict(lambda: [0, 0, 0, 0, 0])   # code -> [title, org, body, core, org_only]
    name_counter: Counter = Counter()
    stats = Counter()
    candidates = []
    for title, body, org in zip(d["제목"].tolist(), d["본문"].tolist(), d["기관"].tolist()):
        title_n = norm_text(title)
        if any(k in title_n for k in MARKET_REPORT_KEYWORDS):
            stats["dropped_market_keyword"] += 1
            continue
        title_n = BROKER_TAIL.sub("", title_n)
        t_names = find_names(title_n, month_alias)
        o_names = {n for n in (norm_text(tag.strip()) for tag in org.split(",")) if n in month_alias}
        t_codes = {month_alias[n] for n in t_names}
        o_codes = {month_alias[n] for n in o_names}
        core_codes = t_codes | o_codes
        if len(core_codes) >= MAX_CODES_PER_ARTICLE:
            stats["dropped_many_codes"] += 1
            continue
        b_names = find_names(norm_text(body), month_alias)
        b_codes = {month_alias[n] for n in b_names}
        all_codes = core_codes | b_codes
        if not all_codes:
            continue
        stats["matched_articles"] += 1
        for c in all_codes:
            row = counts[c]
            row[0] += c in t_codes
            row[1] += c in o_codes
            row[2] += c in b_codes
            row[3] += c in core_codes
            row[4] += c in o_codes and c not in t_codes
        name_counter.update(t_names | o_names)
        if t_names:
            kind, names = "title", t_names
        elif o_names:
            kind, names = "org_only", o_names
        else:
            kind, names = "body_only", b_names
        candidates.append((title[:90], sorted(names), sorted({month_alias[n] for n in names}), kind))
    stats["articles"] = len(d)
    rng = random.Random(zlib.crc32(path.name.encode()))
    samples = rng.sample(candidates, min(SAMPLES_PER_FILE, len(candidates)))
    return month, stats, dict(counts), name_counter, samples


def stage_link() -> None:
    header("[link] 기사 ↔ 종목 연결 (v2 규칙)")
    alias, alias_table = build_alias()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    alias_table.to_csv(RESULTS_DIR / "alias_table.csv", index=False)
    pattern = build_pattern(alias)
    print(f"종목명 {len(alias)}개(정규화 후), 길이 분포: "
          f"{alias_table['name_len'].value_counts().sort_index().head(6).to_dict()}")
    print(f"이름 하나가 여러 종목에 걸리는 경우: {(alias_table['n_codes'] > 1).sum()}개 "
          f"(그 달 모호한 이름은 매칭에서 제외)")

    files = sorted(
        str(p) for p in NEWS_DIR.glob("bigkinds_*.csv.gz")
        if MONTH_START <= p.stem.replace("bigkinds_", "")[:7] <= MONTH_END
    )
    print(f"기사 파일 {len(files)}개, 워커 {N_WORKERS}개")

    counts_all: dict = defaultdict(lambda: [0] * len(COUNT_COLS))
    month_total: Counter = Counter()
    month_stats: dict = defaultdict(Counter)
    name_counts: Counter = Counter()
    samples_all = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(pattern, alias)) as pool:
        for month, stats, counts, name_counter, samples in tqdm(
            pool.map(process_file, files, chunksize=8), total=len(files), desc="news files"
        ):
            month_total[month] += stats["articles"]
            month_stats[month].update(stats)
            name_counts.update(name_counter)
            for code, row in counts.items():
                acc = counts_all[(month, code)]
                for k in range(len(COUNT_COLS)):
                    acc[k] += row[k]
            samples_all.extend((month, *s) for s in samples)

    out = pd.DataFrame(
        [(m, c, *row) for (m, c), row in counts_all.items()],
        columns=["month", "code"] + [f"n_{s}" for s in COUNT_COLS],
    ).sort_values(["month", "code"])
    out.to_csv(COUNTS_PATH, index=False)
    pd.Series(month_total, name="n_articles").rename_axis("month").sort_index().to_csv(
        RESULTS_DIR / "month_article_totals.csv")
    filter_stats = pd.DataFrame(month_stats).T.fillna(0).astype(int).sort_index().rename_axis("month")
    filter_stats.to_csv(RESULTS_DIR / "filter_stats.csv")
    pd.DataFrame(name_counts.most_common(), columns=["name", "n_articles"]).to_csv(
        RESULTS_DIR / "name_match_counts.csv", index=False)
    tot = filter_stats.sum()
    print(f"전처리 후 기사 {tot['articles']:,}건 중 종목 연결 {tot.get('matched_articles', 0):,}건, "
          f"시황 키워드 제외 {tot.get('dropped_market_keyword', 0):,}건, "
          f"여러 종목 나열 제외 {tot.get('dropped_many_codes', 0):,}건")

    # 매칭 정확도 점검용 표본: (매칭 유형 × 이름 길이)별 무작위. 직접 correct(1/0)를 채우면 정밀도 계산
    rows = [{"month": month, "title": title, "matched_names": "|".join(names), "codes": "|".join(codes),
             "kind": kind, "name_len": "3" if min(len(n) for n in names) <= 3 else "4+", "correct": ""}
            for month, title, names, codes, kind in samples_all]
    sample_df = pd.DataFrame(rows)
    picked = pd.concat([
        g.sample(min(SAMPLES_PER_STRATUM, len(g)), random_state=0) for _, g in sample_df.groupby(["kind", "name_len"])
    ])
    picked.sort_values(["kind", "name_len", "month"]).to_csv(RESULTS_DIR / "match_samples.csv", index=False)
    print(f"저장 완료 -> {COUNTS_PATH} (종목×월 {len(out):,}행), results/ 표본·사전")


# ===========================================================================
# 단계 2: 커버리지 리포트
# ===========================================================================

def _setup_korean_font() -> str | None:
    for path in font_manager.findSystemFonts():
        if any(k in path.lower() for k in ("nanum", "cjk", "malgun")):
            font_manager.fontManager.addfont(path)
    installed = {f.name for f in font_manager.fontManager.ttflist}
    name = next((c for c in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic") if c in installed), None)
    if name:
        plt.rcParams["font.family"] = name
        plt.rcParams["axes.unicode_minus"] = False
    return name


def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2)


def load_stock_months() -> pd.DataFrame:
    """월말 시가총액이 있는 종목×월 + 기사 수(없으면 0) + 그 달 시가총액 3분위."""
    mcap = pd.read_csv(RAW / "returns" / "train_2001_2021__raw_month_end_market_cap.csv", dtype={"month": str})
    mcap = mcap.set_index("month")
    mcap.columns = mcap.columns.astype(str)
    mcap = mcap[(mcap.index >= MONTH_START) & (mcap.index <= MONTH_END)]
    long = mcap.stack().rename("mcap").reset_index()
    long.columns = ["month", "code", "mcap"]
    long = long[(long["mcap"] > 0) & ~long["code"].isin(EXCLUDED_CODES)]
    pct = long.groupby("month")["mcap"].rank(pct=True)
    long["size"] = pd.cut(pct, [0, 1 / 3, 2 / 3, 1.0], labels=SIZE_LABELS, include_lowest=True)
    counts = pd.read_csv(COUNTS_PATH, dtype={"month": str, "code": str})
    panel = long.merge(counts, on=["month", "code"], how="left")
    for s in COUNT_COLS:
        if f"n_{s}" in panel:   # 이전 집계 파일에는 n_org_only가 없을 수 있음(link 재실행 전)
            panel[f"n_{s}"] = panel[f"n_{s}"].fillna(0).astype(int)
    panel["year"] = panel["month"].str[:4].astype(int)
    return panel


def report_labeled_precision() -> None:
    """match_samples.csv를 복사해 `match_samples_labeled.csv`로 저장하고 correct 열(1=해당 종목 기사, 0=오탐)을
    채우면 매칭 유형·이름 길이별 정밀도를 계산한다."""
    if not LABELED_SAMPLES.exists():
        print(f"\n[정밀도] {LABELED_SAMPLES.name} 없음 → results/match_samples.csv를 복사해 correct(1/0)를 채우면 계산됨")
        return
    s = pd.read_csv(LABELED_SAMPLES, dtype={"name_len": str})
    s["correct"] = pd.to_numeric(s["correct"], errors="coerce")
    s = s.dropna(subset=["correct"])
    if s.empty:
        print("\n[정밀도] correct 열이 비어 있음")
        return
    prec = s.groupby(["kind", "name_len"])["correct"].agg(precision="mean", n="size").reset_index()
    prec.to_csv(RESULTS_DIR / "precision_by_stratum.csv", index=False)
    print("\n라벨링한 표본 기준 정밀도:")
    print(prec.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    core = s[s["kind"].isin(["title", "org_only"])]
    print(f"제목·기관 태그 매칭 전체 정밀도: {core['correct'].mean():.3f} (n={len(core)})")


def stage_report() -> None:
    header("[report] 커버리지 리포트")
    ko = _setup_korean_font()
    panel = load_stock_months()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    size_name = {"small": "소형", "mid": "중형", "large": "대형"} if ko else {"small": "small", "mid": "mid", "large": "large"}

    # 전체 요약: 소스·시총 구간별 "기사 1건 이상 연결된 종목×월 비율"
    rows = []
    for size in SIZE_LABELS + ["all"]:
        sub = panel if size == "all" else panel[panel["size"] == size]
        row = {"size": size, "n_stock_months": len(sub)}
        for s in SOURCES:
            row[f"share_{s}_ge1"] = (sub[f"n_{s}"] >= 1).mean()
        row["share_core_ge3"] = (sub["n_core"] >= 3).mean()
        row["share_core_ge10"] = (sub["n_core"] >= 10).mean()
        row["mean_n_core"] = sub["n_core"].mean()
        row["median_n_core"] = sub["n_core"].median()
        rows.append(row)
    overall = pd.DataFrame(rows)
    overall.to_csv(RESULTS_DIR / "coverage_overall.csv", index=False)
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    old = RESULTS_DIR / "v1_initial" / "coverage_overall.csv"
    if old.exists():
        o = pd.read_csv(old).set_index("size")
        print("\nv1(이전 규칙) 대비 '기사 1건 이상' 비율(합집합):")
        for size in SIZE_LABELS + ["all"]:
            new = overall.set_index("size").loc[size, "share_core_ge1"]
            print(f"  {size:6s} v1(제목+기관+본문) {o.loc[size, 'share_any_ge1']:.3f} → v2(제목+기관) {new:.3f}")

    # 연도 × 시총 구간
    g = panel.groupby(["year", "size"], observed=True)
    by_year = pd.DataFrame({
        "n_stock_months": g.size(),
        "share_core_ge1": g["n_core"].apply(lambda s: (s >= 1).mean()),
        "share_core_ge3": g["n_core"].apply(lambda s: (s >= 3).mean()),
        "share_title_ge1": g["n_title"].apply(lambda s: (s >= 1).mean()),
        "mean_n_core": g["n_core"].mean(),
    }).reset_index()
    by_year.to_csv(RESULTS_DIR / "coverage_by_year_size.csv", index=False)

    totals = pd.read_csv(RESULTS_DIR / "month_article_totals.csv", dtype={"month": str})
    totals["year"] = totals["month"].str[:4].astype(int)
    print("\n연도별 월평균 전처리 후 기사 수:")
    print(totals.groupby("year")["n_articles"].mean().round(0).astype(int).to_string())

    names = pd.read_csv(RESULTS_DIR / "name_match_counts.csv")
    alias_table = pd.read_csv(RESULTS_DIR / "alias_table.csv", dtype={"name": str})
    top = names.head(60).merge(alias_table[["name", "name_len", "n_codes", "codes"]], on="name", how="left")
    top["share_of_all_matches"] = top["n_articles"] / names["n_articles"].sum()
    top.to_csv(RESULTS_DIR / "top_matched_names.csv", index=False)
    print("\n가장 많이 매칭된 종목명 상위 15(제목·기관 태그 기준, 일반명사·그룹명 오탐 점검용):")
    print(top.head(15)[["name", "n_articles", "name_len", "codes"]].to_string(index=False))

    report_labeled_precision()

    # 플롯 1: 연도별 커버리지 (기사 1건 이상 / 3건 이상), 시총 구간별
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, col, title in zip(
        axes, ["share_core_ge1", "share_core_ge3"],
        ["제목·기관 태그로 기사 1건 이상 연결된 종목×월 비율" if ko else "Share of stock-months with ≥1 article (title/org)",
         "기사 3건 이상" if ko else "≥3 articles"],
    ):
        for size in SIZE_LABELS:
            d = by_year[by_year["size"] == size]
            ax.plot(d["year"], d[col], color=SIZE_COLORS[size], lw=2, marker="o", ms=4, label=size_name[size])
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(2001, 2022, 4))
        _style(ax)
    axes[0].legend(frameon=False, fontsize=10, loc="lower right",
                   title="시가총액 3분위" if ko else "Market-cap tercile")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "coverage_by_year.png", dpi=200)
    plt.close(fig)

    # 플롯 2: 어느 필드가 커버리지를 만드는가 (제목 / 기관 태그 / 본문 / 제목+기관)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor("white")
    labels = SOURCE_LABEL_KO if ko else SOURCE_LABEL_EN
    width = 0.26
    for i, size in enumerate(SIZE_LABELS):
        row = overall[overall["size"] == size].iloc[0]
        vals = [row[f"share_{s}_ge1"] for s in SOURCES]
        x = np.arange(len(SOURCES)) + (i - 1) * width
        ax.bar(x, vals, width * 0.92, color=SIZE_COLORS[size], label=size_name[size])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(SOURCES)), [labels[s] for s in SOURCES], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("매칭 필드별 커버리지 (기사 1건 이상 연결된 종목×월 비율, 2001-2021)" if ko
                 else "Coverage by matching field (share of stock-months with ≥1 article, 2001-2021)",
                 loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "coverage_by_field.png", dpi=200)
    plt.close(fig)
    print(f"\n이미지 저장 완료 -> {IMAGES_DIR}")


def stage_variant_plots() -> None:
    """매칭 방식 세 가지(제목+기관+본문 / 제목+기관 태그 / 기관 태그로만)의 연도별 매칭률을 한 도표에 비교.
    저장된 집계 CSV만 사용하고, 시총 3분위는 구간별 종목×월 수로 가중평균해 전체 종목 기준으로 합친다.
    기관만 = (제목+기관 태그) − 제목: 제목 매칭 없이 기관 태그로만 걸린 종목×월."""
    header("[variantplots] 매칭 방식별 연도별 매칭률 비교")
    ko = _setup_korean_font()
    v1 = pd.read_csv(RESULTS_DIR / "v1_initial" / "coverage_by_year_size.csv")
    v2 = pd.read_csv(RESULTS_DIR / "coverage_by_year_size.csv")
    v2["share_org_only_ge1"] = v2["share_core_ge1"] - v2["share_title_ge1"]

    def yearly(df: pd.DataFrame, col: str) -> pd.Series:
        w = df["n_stock_months"]
        return (df[col] * w).groupby(df["year"]).sum() / w.groupby(df["year"]).sum()

    series = [
        ("제목+기관+본문" if ko else "title + org tag + body", yearly(v1, "share_any_ge1"), "#2a78d6"),
        ("제목+기관 태그" if ko else "title + org tag", yearly(v2, "share_core_ge1"), "#eb6834"),
        ("기관 태그로만 (제목 매칭 없음)" if ko else "org tag only (no title match)", yearly(v2, "share_org_only_ge1"), "#1baf7a"),
    ]
    out_dir = IMAGES_DIR / "variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("white")
    for label, ys, color in series:
        ax.plot(ys.index, ys.to_numpy(), color=color, lw=2, marker="o", ms=4, label=label)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(2001, 2022, 2))
    ax.set_ylabel("기사 1건 이상 연결된 종목×월 비율" if ko else "Share of stock-months with ≥1 article", color=INK2)
    ax.set_title("매칭 방식별 연도별 매칭률 (전체 종목)" if ko else "Matching rate by method (all stocks)",
                 loc="left", fontsize=13, color=INK)
    ax.legend(frameon=False, fontsize=10, loc="upper left", title="매칭 방식" if ko else "Matching method")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_compare.png", dpi=200)
    plt.close(fig)
    pd.DataFrame({label: ys for label, ys, _ in series}).rename_axis("year").to_csv(
        RESULTS_DIR / "coverage_by_year_variant.csv")
    print(f"이미지 저장 완료 -> {out_dir}/coverage_compare.png")


def stage_report_v1() -> None:
    """results/v1_initial/의 v1(이전 규칙) 집계 CSV로 커버리지 도표를 다시 그려 images/v1_initial/에 저장."""
    header("[v1plots] v1(이전 규칙) 커버리지 도표")
    v1 = RESULTS_DIR / "v1_initial"
    ko = _setup_korean_font()
    overall = pd.read_csv(v1 / "coverage_overall.csv")
    by_year = pd.read_csv(v1 / "coverage_by_year_size.csv")
    out_dir = IMAGES_DIR / "v1_initial"
    out_dir.mkdir(parents=True, exist_ok=True)
    size_name = {"small": "소형", "mid": "중형", "large": "대형"} if ko else {"small": "small", "mid": "mid", "large": "large"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, col, title in zip(
        axes, ["share_any_ge1", "share_any_ge3"],
        ["기사 1건 이상 연결된 종목×월 비율 (제목+기관+본문)" if ko
         else "Share of stock-months with ≥1 article (title+org+body)",
         "기사 3건 이상" if ko else "≥3 articles"],
    ):
        for size in SIZE_LABELS:
            d = by_year[by_year["size"] == size]
            ax.plot(d["year"], d[col], color=SIZE_COLORS[size], lw=2, marker="o", ms=4, label=size_name[size])
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(2001, 2022, 4))
        _style(ax)
    axes[0].legend(frameon=False, fontsize=10, loc="lower right",
                   title="시가총액 3분위" if ko else "Market-cap tercile")
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_by_year.png", dpi=200)
    plt.close(fig)

    v1_sources = ["title", "org", "body", "any"]
    labels = ({"title": "제목", "org": "기관 태그", "body": "본문(200자)", "any": "합집합\n(제목+기관+본문)"} if ko
              else {"title": "title", "org": "org tag", "body": "body (200 chars)", "any": "union\n(title+org+body)"})
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor("white")
    width = 0.26
    for i, size in enumerate(SIZE_LABELS):
        row = overall[overall["size"] == size].iloc[0]
        vals = [row[f"share_{s}_ge1"] for s in v1_sources]
        x = np.arange(len(v1_sources)) + (i - 1) * width
        ax.bar(x, vals, width * 0.92, color=SIZE_COLORS[size], label=size_name[size])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(v1_sources)), [labels[s] for s in v1_sources], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("매칭 필드별 커버리지 (기사 1건 이상 연결된 종목×월 비율, 2001-2021)" if ko
                 else "Coverage by matching field (share of stock-months with ≥1 article, 2001-2021)",
                 loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_by_field.png", dpi=200)
    plt.close(fig)
    print(f"이미지 저장 완료 -> {out_dir}")


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    if "v1plots" in sys.argv[1:]:
        stage_report_v1()
        return
    if "variantplots" in sys.argv[1:]:
        stage_variant_plots()
        return
    if "report" not in sys.argv[1:]:
        stage_link()
    stage_report()


if __name__ == "__main__":
    main()
