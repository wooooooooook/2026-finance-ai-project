"""당월(가장 최근 실현) 수익률 1개만 피처로 써서 XGBoost로 다음달 수익률을 예측하는 가장
단순한 나이브 모델. step2(수익률 lag 1/3/6/12개월 4개)보다 더 단순한 버전 — 여러 lag
윈도우를 쓰지 않고 가장 최근 값 하나만 본다.
"""

from pathlib import Path

import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
FEATURE_COLS = ["return_lag1"]
TRAIN_YEARS = ("2001-01", "2017-12")
VAL_YEARS = ("2018-01", "2021-12")

df = pd.read_csv(BASE_DIR / "features" / "return_features.csv", dtype={"code": str})
df = df.dropna(subset=["target_next"])

train_df = df[(df["month"] >= TRAIN_YEARS[0]) & (df["month"] <= TRAIN_YEARS[1])]
val_df = df[(df["month"] >= VAL_YEARS[0]) & (df["month"] <= VAL_YEARS[1])]
print(f"train={len(train_df)}, val={len(val_df)}")

model = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, missing=float("nan"))
model.fit(train_df[FEATURE_COLS], train_df["target_next"])

pred = model.predict(val_df[FEATURE_COLS])
rmse = ((pred - val_df["target_next"]) ** 2).mean() ** 0.5
mae = (pred - val_df["target_next"]).abs().mean()
print(f"RMSE: {rmse:.6f}, MAE: {mae:.6f}")

tmp = pd.DataFrame({"month": val_df["month"].to_numpy(), "pred": pred, "actual": val_df["target_next"].to_numpy()})
ic = tmp.groupby("month").apply(lambda g: g["pred"].corr(g["actual"], method="spearman"), include_groups=False).dropna()
print(f"Rank IC 평균: {ic.mean():.4f}, 표준편차: {ic.std():.4f}, IC-IR: {ic.mean()/ic.std():.4f}, "
      f"양의 IC 비율: {(ic > 0).mean():.1%} ({len(ic)}개월)")
