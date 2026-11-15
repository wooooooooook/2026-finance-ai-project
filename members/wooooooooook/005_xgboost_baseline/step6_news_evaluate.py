"""
step6_news_train_model.py가 저장한 모델 weight를 불러와 validation(기본)/test에서
성능을 평가한다. baseline(step4, 매매동향+매크로)에 뉴스 피처(일자가중 감성점수 3종)를
추가한 버전. 학습(step6_news_train_model.py)과 평가를 분리해서, 모델을 다시 학습하지
않고도 평가만 반복할 수 있다.

test는 최종 평가 전 하이퍼파라미터 선택 등에 쓰면 안 되므로, 기본값은 validation이고
test는 --split test로 명시했을 때만 돈다.

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step6_news_evaluate.py              # validation 평가
    python3 step6_news_evaluate.py --split test  # 최종 test 평가 (모델/하이퍼파라미터 다 정해진 뒤에만)
"""

import argparse

import pandas as pd
import xgboost as xgb

from step6_news_train_model import (
    FEATURE_COLS,
    MODEL_PATH,
    TEST_YEARS,
    VAL_YEARS,
    load_merged_features,
)


def _monthly_ic(split_df: pd.DataFrame, pred) -> pd.Series:
    # 월별 횡단면(그 달 전 종목) 예측값-실제값 상관계수 (Rank IC, 스피어만)
    tmp = pd.DataFrame({
        "month": split_df["month"].to_numpy(),
        "pred": pred,
        "actual": split_df["target_next"].to_numpy(),
    })
    return tmp.groupby("month").apply(
        lambda g: g["pred"].corr(g["actual"], method="spearman"), include_groups=False,
    )


def evaluate(split: str = "val"):
    period = VAL_YEARS if split == "val" else TEST_YEARS

    df = load_merged_features()
    split_df = df[(df["month"] >= period[0]) & (df["month"] <= period[1])]
    print(f"[{split}] 평가 대상 행수: {len(split_df)} ({period[0]} ~ {period[1]})")

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    pred = model.predict(split_df[FEATURE_COLS])
    rmse = ((pred - split_df["target_next"]) ** 2).mean() ** 0.5
    mae = (pred - split_df["target_next"]).abs().mean()
    print(f"[{split}] RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    ic = _monthly_ic(split_df, pred).dropna()
    ic_mean = ic.mean()
    ic_std = ic.std()
    ic_ir = ic_mean / ic_std if ic_std else float("nan")
    hit_rate = (ic > 0).mean()
    print(f"[{split}] Rank IC 평균: {ic_mean:.4f}, 표준편차: {ic_std:.4f}, "
          f"IC-IR: {ic_ir:.4f}, 양의 IC 비율: {hit_rate:.1%} ({len(ic)}개월)")

    return rmse, mae, ic_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    if args.split == "test":
        print("주의: test는 최종 평가 전용입니다. 모델/하이퍼파라미터가 다 정해진 뒤에만 실행하세요.")

    evaluate(args.split)


if __name__ == "__main__":
    main()
