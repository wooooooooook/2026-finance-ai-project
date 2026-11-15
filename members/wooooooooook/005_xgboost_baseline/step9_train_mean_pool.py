"""
step7_build_news_embeddings.py가 저장한 기사별 768차원 FinBERT 임베딩을 재사용해서,
attention(학습된 가중치) 없이 **단순 평균**으로 월별 피처를 만든다. step7과의 유일한 차이는
"가중치를 학습하느냐 안 하느냐"뿐 — attention의 효과를 떼어내 보기 위한 ablation.

구조(Deep Sets 방식 — 기사별로 먼저 비선형 변환한 뒤 평균):
  기사 임베딩(768) -> project(768->32->4, Tanh) -> 기사별 4차원 벡터
  -> 그 달 기사들에 대해 단순 평균(가중치 없음, 전부 1/n) -> 4차원 월별 벡터 [최종 뉴스 피처]
  -> (학습 시에만) head(4->1) -> 다음달 수익률 예측, MSE loss

project와 head만 학습 대상이고 FinBERT는 완전히 고정(이미 임베딩만 캐싱해둠). 학습은 train
기간 월만 사용(리크 방지), validation 기간으로 매 epoch loss 모니터링하면서 가장 좋은
가중치를 저장한다(단순 early stopping, step7/step8과 동일 방식).

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step9_train_mean_pool.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from step3_baseline_train_model import TEST_YEARS, TRAIN_YEARS, VAL_YEARS
from step9_news_train_model import RETURN_FEATURES_PATH, build_return_features

BASE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = BASE_DIR / "features"
EMBEDDINGS_PATH = FEATURES_DIR / "news_embeddings.npz"
OUT_PATH = FEATURES_DIR / "news_features_meanpool_step9.csv"

PROJECT_DIM = 4
EPOCHS = 30
LR = 1e-3
SEED = 42


class MeanPoolProjector(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 32, out_dim: int = PROJECT_DIM):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, out_dim))
        self.head = nn.Linear(out_dim, 1)

    def forward(self, month_embeddings: torch.Tensor):
        projected = self.project(month_embeddings)   # [n, PROJECT_DIM] — 기사별 독립 투영
        pooled = projected.mean(dim=0)                 # [PROJECT_DIM] — attention 없이 단순 평균
        pred = self.head(pooled)                       # [1]
        return pooled, pred


def load_embeddings_by_month() -> dict[str, torch.Tensor]:
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    month, embeddings = data["month"], data["embeddings"]
    out = {}
    for m in np.unique(month):
        out[str(m)] = torch.tensor(embeddings[month == m], dtype=torch.float32)
    return out


def load_targets_by_month() -> dict[str, torch.Tensor]:
    if not RETURN_FEATURES_PATH.exists():
        build_return_features()
    returns = pd.read_csv(RETURN_FEATURES_PATH, dtype={"code": str})
    returns = returns.dropna(subset=["target_next"])
    out = {}
    for month, g in returns.groupby("month"):
        out[month] = torch.tensor(g["target_next"].to_numpy(), dtype=torch.float32)
    return out


def _months_in(period) -> list[str]:
    start, end = period
    return [m for m in sorted({f"{y:04d}-{mo:02d}" for y in range(2000, 2026) for mo in range(1, 13)})
            if start <= m <= end]


def _epoch_loss(model, months, embeddings_by_month, targets_by_month, optimizer=None):
    total_loss, n = 0.0, 0
    for month in months:
        if month not in embeddings_by_month or month not in targets_by_month:
            continue
        emb = embeddings_by_month[month]
        targets = targets_by_month[month]

        _, pred = model(emb)
        loss = ((pred - targets) ** 2).mean()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n if n else float("nan")


def train() -> MeanPoolProjector:
    torch.manual_seed(SEED)

    embeddings_by_month = load_embeddings_by_month()
    targets_by_month = load_targets_by_month()

    train_months = _months_in(TRAIN_YEARS)
    val_months = _months_in(VAL_YEARS)

    model = MeanPoolProjector()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None

    for epoch in tqdm(range(EPOCHS), desc="mean pool 학습 (attention 없음)"):
        model.train()
        train_loss = _epoch_loss(model, train_months, embeddings_by_month, targets_by_month, optimizer)

        model.eval()
        with torch.no_grad():
            val_loss = _epoch_loss(model, val_months, embeddings_by_month, targets_by_month)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        tqdm.write(f"epoch {epoch + 1}/{EPOCHS} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    print(f"최종 선택: val_loss={best_val_loss:.6f} 기준 best epoch 가중치")
    return model


def export_features(model: MeanPoolProjector) -> None:
    embeddings_by_month = load_embeddings_by_month()
    all_months = _months_in(TRAIN_YEARS) + _months_in(VAL_YEARS) + _months_in(TEST_YEARS)

    model.eval()
    rows = []
    with torch.no_grad():
        for month in all_months:
            if month not in embeddings_by_month:
                continue
            pooled, _ = model(embeddings_by_month[month])
            rows.append([month] + pooled.tolist())

    cols = ["month"] + [f"meanpool_news_{i}" for i in range(PROJECT_DIM)]
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(OUT_PATH, index=False)
    print(f"저장 완료 -> {OUT_PATH}, 행수={len(out)}")


def main() -> None:
    model = train()
    export_features(model)


if __name__ == "__main__":
    main()
