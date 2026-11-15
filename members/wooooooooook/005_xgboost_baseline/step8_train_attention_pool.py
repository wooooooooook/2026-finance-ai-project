"""
step8_build_bge_embeddings.py(key: BGE-M3 1024차원)와 step8_build_sentiment_raw.py(value:
FinBERT 긍정/부정/중립 확률 3차원)를 기사 단위로 정렬해서, 월별로 attention 가중합한 3차원
뉴스 피처를 만든다. step7과 달리 "가중치를 계산하는 기준"과 "실제로 집계하는 값"을 분리한
버전 — BGE-M3가 그 기사가 내용적으로 중요한지 판단하고(key), FinBERT는 감성 신호만
제공한다(value). 출력 차원도 step6과 동일한 3차원으로 맞춰서 "집계 방식"만 비교할 수 있게
했다.

구조: key 임베딩(1024) -> score MLP(1024->32->1) -> 월 안에서 softmax
      -> value(3차원, 감성확률)에 그 가중치로 가중합 -> 3차원 월별 벡터 [최종 뉴스 피처]
      -> (학습 시에만) head(3->1) -> 다음달 수익률 예측, MSE loss

학습은 train 기간 월만 사용(리크 방지), validation 기간으로 매 epoch loss 모니터링하면서
가장 좋은 가중치를 저장한다(단순 early stopping, step7과 동일 방식).

사용법:
    cd members/wooooooooook/005_xgboost_baseline
    python3 step8_train_attention_pool.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from step3_baseline_train_model import TEST_YEARS, TRAIN_YEARS, VAL_YEARS
from step8_news_train_model import RETURN_FEATURES_PATH, build_return_features

BASE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = BASE_DIR / "features"
KEY_PATH = FEATURES_DIR / "news_bge_embeddings_step8.npz"
VALUE_PATH = FEATURES_DIR / "news_sentiment_raw_step8.npz"
OUT_PATH = FEATURES_DIR / "news_features_attn_step8.csv"

VALUE_COLS = ["attn_sentiment_positive", "attn_sentiment_negative", "attn_sentiment_neutral"]
EPOCHS = 30
LR = 1e-3
SEED = 42


class KeyValueAttentionPool(nn.Module):
    def __init__(self, key_dim: int = 1024, hidden: int = 32, value_dim: int = 3):
        super().__init__()
        self.score_mlp = nn.Sequential(nn.Linear(key_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Linear(value_dim, 1)

    def forward(self, key_embeddings: torch.Tensor, values: torch.Tensor):
        scores = self.score_mlp(key_embeddings).squeeze(-1)   # [n]
        weights = torch.softmax(scores, dim=0)                 # [n]
        pooled = (weights.unsqueeze(-1) * values).sum(0)        # [value_dim]
        pred = self.head(pooled)                                # [1]
        return pooled, pred


def load_by_month() -> dict[str, dict[str, torch.Tensor]]:
    key_data = np.load(KEY_PATH, allow_pickle=True)
    val_data = np.load(VALUE_PATH, allow_pickle=True)

    # 두 npz 모두 step8_sample_news_pool.py의 같은 캐시를 순서 그대로 읽어서 만들었으니
    # 행 순서가 일치해야 한다 — 혹시 모를 불일치를 바로 잡아낸다.
    assert len(key_data["month"]) == len(val_data["month"]), "key/value 샘플 수가 다름"
    assert (key_data["month"] == val_data["month"]).all(), "key/value month가 안 맞음 (샘플 순서 불일치)"
    assert (key_data["day"] == val_data["day"]).all(), "key/value day가 안 맞음 (샘플 순서 불일치)"

    month = key_data["month"]
    keys = key_data["embeddings"]
    values = np.stack([val_data["p_positive"], val_data["p_negative"], val_data["p_neutral"]], axis=1)

    out = {}
    for m in np.unique(month):
        mask = month == m
        out[str(m)] = {
            "key": torch.tensor(keys[mask], dtype=torch.float32),
            "value": torch.tensor(values[mask], dtype=torch.float32),
        }
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


def _epoch_loss(model, months, by_month, targets_by_month, optimizer=None):
    total_loss, n = 0.0, 0
    for month in months:
        if month not in by_month or month not in targets_by_month:
            continue
        key, value = by_month[month]["key"], by_month[month]["value"]
        targets = targets_by_month[month]

        _, pred = model(key, value)
        loss = ((pred - targets) ** 2).mean()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n if n else float("nan")


def train() -> KeyValueAttentionPool:
    torch.manual_seed(SEED)

    by_month = load_by_month()
    targets_by_month = load_targets_by_month()

    train_months = _months_in(TRAIN_YEARS)
    val_months = _months_in(VAL_YEARS)

    model = KeyValueAttentionPool()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None

    for epoch in tqdm(range(EPOCHS), desc="attention pool 학습 (BGE-M3 key / FinBERT value)"):
        model.train()
        train_loss = _epoch_loss(model, train_months, by_month, targets_by_month, optimizer)

        model.eval()
        with torch.no_grad():
            val_loss = _epoch_loss(model, val_months, by_month, targets_by_month)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        tqdm.write(f"epoch {epoch + 1}/{EPOCHS} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    print(f"최종 선택: val_loss={best_val_loss:.6f} 기준 best epoch 가중치")
    return model


def export_features(model: KeyValueAttentionPool) -> None:
    by_month = load_by_month()
    all_months = _months_in(TRAIN_YEARS) + _months_in(VAL_YEARS) + _months_in(TEST_YEARS)

    model.eval()
    rows = []
    with torch.no_grad():
        for month in all_months:
            if month not in by_month:
                continue
            pooled, _ = model(by_month[month]["key"], by_month[month]["value"])
            rows.append([month] + pooled.tolist())

    out = pd.DataFrame(rows, columns=["month"] + VALUE_COLS)
    out.to_csv(OUT_PATH, index=False)
    print(f"저장 완료 -> {OUT_PATH}, 행수={len(out)}")


def main() -> None:
    model = train()
    export_features(model)


if __name__ == "__main__":
    main()
