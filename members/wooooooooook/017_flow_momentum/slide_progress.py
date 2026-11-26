"""
slide_progress.py: 발표용 도표 — 005 → 006 → 007로 오면서 VAL Rank IC가 어떻게 올라왔는지.
수치는 각 실험 README의 VAL(2018-2021) 결과를 그대로 옮긴 값이다(재계산 아님).

사용법:
    uv run python members/wooooooooook/017_flow_momentum/slide_progress.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

IMAGES_DIR = Path(__file__).resolve().parent / "images"

# (한글 라벨, 영어 라벨, VAL Rank IC, 뉴스 포함 여부)
STEPS = [
    ("005 step3\n매매동향 8개", "005 step3\nflow 8", 0.0365, False),
    ("005 step4\n+ 매크로 (13개)", "005 step4\n+ macro (13)", 0.0416, False),
    ("005 step7\n+ 뉴스 attention", "005 step7\n+ news attention", 0.0540, True),
    ("006\nfeature 23개", "006\n23 features", 0.0869, False),
    ("007\n+ 순위 타깃", "007\n+ rank target", 0.1008, False),
]
COLOR_STRUCT, COLOR_NEWS = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"


def setup_korean_font() -> str | None:
    for path in font_manager.findSystemFonts():
        if any(k in path.lower() for k in ("nanum", "cjk", "malgun")):
            font_manager.fontManager.addfont(path)
    installed = {f.name for f in font_manager.fontManager.ttflist}
    name = next((c for c in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic") if c in installed), None)
    if name:
        plt.rcParams["font.family"] = name
        plt.rcParams["axes.unicode_minus"] = False
    return name


def main() -> None:
    ko = setup_korean_font()
    print(f"폰트: {'한글 폰트 없음 → 영어 라벨 사용' if not ko else ko}")
    labels = [s[0] if ko else s[1] for s in STEPS]
    values = [s[2] for s in STEPS]
    colors = [COLOR_NEWS if s[3] else COLOR_STRUCT for s in STEPS]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.patch.set_facecolor("white")
    bars = ax.bar(range(len(STEPS)), values, 0.55, color=colors)
    for i, (b, v) in enumerate(zip(bars, values)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=11, color=INK)
        if i > 0:
            prev = values[i - 1]
            ax.text(b.get_x() + b.get_width() / 2, v / 2, f"{v - prev:+.4f}\n({'이전 단계 대비' if ko else 'vs prev'})",
                    ha="center", va="center", fontsize=9, color="white")
    ax.set_xticks(range(len(STEPS)), labels, fontsize=10)
    ax.set_ylim(0, 0.12)
    ax.set_ylabel("VAL 2018-2021 월별 Rank IC 평균" if ko else "VAL 2018-2021 mean monthly Rank IC", color=INK2)
    ax.set_title("실험 단계별 Rank IC 개선" if ko else "Rank IC by experiment stage", loc="left", fontsize=13, color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_STRUCT), plt.Rectangle((0, 0), 1, 1, color=COLOR_NEWS)]
    names = ["뉴스 미포함", "뉴스 포함"] if ko else ["without news", "with news"]
    ax.legend(handles, names, frameon=False, fontsize=10, loc="upper left")

    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2)
    note = ("월별 IC 표준편차 ≈ 0.11 → 48개월 평균의 표준오차 ≈ 0.016 (006 README)" if ko
            else "Monthly IC std ≈ 0.11 → SE of 48-month mean ≈ 0.016 (006 README)")
    fig.text(0.01, 0.01, note, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out = IMAGES_DIR / "slide_progress.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"저장 완료 -> {out}")


if __name__ == "__main__":
    main()
