"""
step4_baseline_train_model.py가 저장한 모델(models/xgb_baseline_macro.json)에 대해
SHAP(TreeExplainer)로 각 feature의 기여도를 분석한다. XGBoost 내장 gain-importance와
달리, "그 feature 값이 클 때 예측에 +로 작용했는지 -로 작용했는지" 방향까지 보여준다.

출력:
- 콘솔: feature별 평균 |SHAP| (전역 중요도, 높을수록 예측에 큰 영향) 순위
- 이미지: quicktest/step4_shap_summary_<split>.png (summary beeswarm plot — feature별
  SHAP 분포와 방향을 한 번에 보여줌)

test는 최종 평가 전 하이퍼파라미터 선택 등에 쓰면 안 되므로, 기본값은 validation이고
test는 --split test로 명시했을 때만 돈다.

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step4_shap_importance.py               # validation 기준(기본)
    python3 step4_shap_importance.py --split test   # 최종 test 기준 (모델/하이퍼파라미터 다 정해진 뒤에만)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb

from step4_baseline_train_model import (
    FEATURE_COLS,
    MODEL_PATH,
    TEST_YEARS,
    VAL_YEARS,
    load_merged_features,
)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "quicktest"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    if args.split == "test":
        print("주의: test는 최종 평가 전용입니다. 모델/하이퍼파라미터가 다 정해진 뒤에만 실행하세요.")

    period = VAL_YEARS if args.split == "val" else TEST_YEARS

    df = load_merged_features()
    split_df = df[(df["month"] >= period[0]) & (df["month"] <= period[1])]
    X = split_df[FEATURE_COLS]
    print(f"[{args.split}] SHAP 분석 대상 행수: {len(X)} ({period[0]} ~ {period[1]})")

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranking = sorted(zip(FEATURE_COLS, mean_abs_shap), key=lambda x: -x[1])
    print("\n=== feature별 평균 |SHAP| (전역 중요도, 높을수록 예측에 큰 영향) ===")
    for name, val in ranking:
        print(f"{name:30s} {val:.6f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X, show=False)
    out_path = OUT_DIR / f"step4_shap_summary_{args.split}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nsummary plot 저장 완료 -> {out_path}")


if __name__ == "__main__":
    main()
