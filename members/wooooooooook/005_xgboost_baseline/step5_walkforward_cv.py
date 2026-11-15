"""
step4_baseline_train_model.py의 피처(trading_flow + returns + macro, 13개)를 그대로
재사용해서, 단일 train/val 분할 대신 walk-forward(expanding window) 교차검증으로 baseline
모델을 여러 구간에서 평가한다. 단일 분할 하나만 보면 그 구간이 우연히 좋거나 나쁘게 나올
위험이 있어서, 여러 fold에 걸친 평균/구간별 편차를 같이 본다.

fold 구성: 2001-2008(8년)을 최초 train으로 고정하고, 그 뒤를 val 윈도우 크기(--val-years,
기본 4년)만큼씩 잘라 expanding window로 진행한다. 마지막 fold는 2021-12에서 끝나도록 길이를
맞춘다. --val-years 1로 주면 1년 단위로 더 잘게(약 13개 fold) 돌려볼 수 있다.

test(2022-2025)는 여기서 쓰지 않는다 — 최종 평가 전용.

미리 준비돼 있어야 하는 것: features/trading_flow_features.csv, features/return_features.csv
(step4_baseline_train_model.py를 한 번 실행하면 생성됨). 이 스크립트는 그걸 다시 만들지
않고 읽기만 한다.

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step5_walkforward_cv.py                  # 4년 단위 (기본, fold 3개)
    python3 step5_walkforward_cv.py --val-years 1     # 1년 단위 (fold 약 13개)
"""

import argparse
from pathlib import Path

import pandas as pd
import xgboost as xgb

from step4_baseline_train_model import FEATURE_COLS, load_merged_features

BASE_DIR = Path(__file__).resolve().parent

INITIAL_TRAIN_END = "2008-12"
OVERALL_END = "2021-12"


def _add_months(ym: str, months: int) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    total = y * 12 + (m - 1) + months
    ny, nm = divmod(total, 12)
    return f"{ny:04d}-{nm + 1:02d}"


def make_folds(val_years: int) -> list[dict]:
    val_months = val_years * 12
    folds = []
    train_end = INITIAL_TRAIN_END
    val_start = _add_months(train_end, 1)
    while val_start <= OVERALL_END:
        val_end = min(_add_months(val_start, val_months - 1), OVERALL_END)
        folds.append({"train": ("2001-01", train_end), "val": (val_start, val_end)})
        train_end = val_end
        val_start = _add_months(train_end, 1)
    return folds


def _monthly_ic(split_df: pd.DataFrame, pred) -> pd.Series:
    tmp = pd.DataFrame({
        "month": split_df["month"].to_numpy(),
        "pred": pred,
        "actual": split_df["target_next"].to_numpy(),
    })
    return tmp.groupby("month").apply(
        lambda g: g["pred"].corr(g["actual"], method="spearman"), include_groups=False,
    )


def run_fold(df: pd.DataFrame, fold: dict, fold_idx: int) -> dict:
    train_start, train_end = fold["train"]
    val_start, val_end = fold["val"]

    train_df = df[(df["month"] >= train_start) & (df["month"] <= train_end)]
    val_df = df[(df["month"] >= val_start) & (df["month"] <= val_end)]
    print(f"[fold {fold_idx}] train={train_start}~{train_end}({len(train_df)}행), "
          f"val={val_start}~{val_end}({len(val_df)}행)")

    model = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, missing=float("nan"))
    model.fit(train_df[FEATURE_COLS], train_df["target_next"])

    pred = model.predict(val_df[FEATURE_COLS])
    rmse = ((pred - val_df["target_next"]) ** 2).mean() ** 0.5
    mae = (pred - val_df["target_next"]).abs().mean()

    ic = _monthly_ic(val_df, pred).dropna()
    ic_mean = ic.mean()
    ic_std = ic.std()
    ic_ir = ic_mean / ic_std if ic_std else float("nan")
    hit_rate = (ic > 0).mean()

    print(f"[fold {fold_idx}] RMSE={rmse:.6f}, MAE={mae:.6f}, IC평균={ic_mean:.4f}, "
          f"IC표준편차={ic_std:.4f}, IC-IR={ic_ir:.4f}, 양의 IC 비율={hit_rate:.1%} ({len(ic)}개월)")

    return {
        "fold": fold_idx, "rmse": rmse, "mae": mae,
        "ic_mean": ic_mean, "ic_std": ic_std, "ic_ir": ic_ir, "hit_rate": hit_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-years", type=int, default=4, help="fold당 validation 기간(년)")
    args = parser.parse_args()

    folds = make_folds(args.val_years)
    print(f"val-years={args.val_years}, fold 수={len(folds)}")

    df = load_merged_features()

    results = [run_fold(df, fold, i + 1) for i, fold in enumerate(folds)]
    summary = pd.DataFrame(results)

    print("\n=== fold 평균 (walk-forward CV 요약) ===")
    print(summary[["rmse", "mae", "ic_mean", "ic_ir", "hit_rate"]].mean().to_string())
    print("\n=== fold별 IC 평균 편차(표준편차) — fold 간 변동이 크면 단일 분할로는 불안정하다는 뜻 ===")
    print(f"fold 간 IC 평균의 표준편차: {summary['ic_mean'].std():.4f}")


if __name__ == "__main__":
    main()
