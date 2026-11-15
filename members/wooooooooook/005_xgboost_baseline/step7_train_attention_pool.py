"""
step7_build_news_embeddings.py가 저장한 기사별 768차원 임베딩을, 월별로 attention(학습된
가중치)으로 가중합해서 8차원 뉴스 피처를 만든다. FinBERT는 완전히 고정돼 있고(이미 임베딩만
캐싱해둠), 여기서 학습하는 건 attention score와 8차원 투영을 담당하는 작은 MLP뿐이다.

구조: 기사 임베딩(768) -> score MLP(768->32->1) -> 월 안에서 softmax -> 가중합(768)
      -> 투영(768->8)  [이 8차원이 최종 뉴스 피처]
      -> (학습 시에만) head(8->1) -> 다음달 수익률 예측, MSE loss

학습은 train 기간 월만 사용(리크 방지), validation 기간으로 매 epoch loss 모니터링하면서
가장 좋은 가중치를 저장한다(단순 early stopping). 학습이 끝나면 train+val+test 전체 월에
대해 8차원 피처를 뽑아 features/news_features_mlp.csv로 저장한다.

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step7_train_attention_pool.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from step3_baseline_train_model import TEST_YEARS, TRAIN_YEARS, VAL_YEARS
from step7_news_train_model import RETURN_FEATURES_PATH, build_return_features

BASE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = BASE_DIR / "features"
EMBEDDINGS_PATH = FEATURES_DIR / "news_embeddings.npz"
OUT_PATH = FEATURES_DIR / "news_features_mlp.csv"

PROJECT_DIM = 8
EPOCHS = 30
LR = 1e-3
SEED = 42


class AttentionPool(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 32, out_dim: int = PROJECT_DIM):
        super().__init__()
        self.score_mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.project = nn.Linear(in_dim, out_dim)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, month_embeddings: torch.Tensor):
        scores = self.score_mlp(month_embeddings).squeeze(-1)      # [n]
        weights = torch.softmax(scores, dim=0)                      # [n]
        pooled = (weights.unsqueeze(-1) * month_embeddings).sum(0)   # [768]
        projected = self.project(pooled)                            # [PROJECT_DIM]
        pred = self.head(projected)                                 # [1]
        return projected, pred


def load_embeddings_by_month() -> dict[str, torch.Tensor]:
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    month, embeddings = data["month"], data["embeddings"]
    out = {}
    for m in np.unique(month):
        out[str(m)] = torch.tensor(embeddings[month == m], dtype=torch.float32)
    return out


def load_targets_by_month() -> dict[str, torch.Tensor]:
    # step6가 동시에 features/return_features.csv를 다시 쓰고 있을 수 있어서, step7
    # 전용 복사본(RETURN_FEATURES_PATH)을 직접 만들어 쓴다 (없으면 지금 생성).
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
        loss = ((pred - targets) ** 2).mean()  # pred(스칼라)를 그 달 전 종목 타겟에 broadcast

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n if n else float("nan")


def train() -> AttentionPool:
    torch.manual_seed(SEED)

    embeddings_by_month = load_embeddings_by_month()
    targets_by_month = load_targets_by_month()

    train_months = _months_in(TRAIN_YEARS)
    val_months = _months_in(VAL_YEARS)

    model = AttentionPool()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None

    for epoch in tqdm(range(EPOCHS), desc="attention pool 학습"):
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


def export_features(model: AttentionPool) -> None:
    embeddings_by_month = load_embeddings_by_month()
    all_months = _months_in(TRAIN_YEARS) + _months_in(VAL_YEARS) + _months_in(TEST_YEARS)

    model.eval()
    rows = []
    with torch.no_grad():
        for month in all_months:
            if month not in embeddings_by_month:
                continue
            projected, _ = model(embeddings_by_month[month])
            rows.append([month] + projected.tolist())

    cols = ["month"] + [f"mlp_news_{i}" for i in range(PROJECT_DIM)]
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(OUT_PATH, index=False)
    print(f"저장 완료 -> {OUT_PATH}, 행수={len(out)}")


def main() -> None:
    model = train()
    export_features(model)


if __name__ == "__main__":
    main()
