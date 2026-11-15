"""
2001년~2005년 각 월 15일자 bigkinds xlsx 파일의 '제목' 컬럼을
snunlp/KR-SBERT-V40K-klueNLI-augSTS 모델(문장 의미 임베딩 전용, mean pooling)로
768차원 임베딩으로 변환해 저장한다. 감성분류 모델(KR-FinBert-SC) 대신 사용.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
DOWNLOADS_DIR = Path(__file__).parent.parent / "bigkinds-crawler" / "downloads"
OUTPUT_DIR = Path(__file__).parent / "embeddings"
YEARS = range(2001, 2006)
BATCH_SIZE = 64


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    targets = [
        DOWNLOADS_DIR / f"bigkinds_{year}-{month:02d}-15.xlsx"
        for year in YEARS
        for month in range(1, 13)
    ]

    for xlsx_path in tqdm(targets, desc="files"):
        if not xlsx_path.exists():
            print(f"skip (missing): {xlsx_path.name}")
            continue

        output_path = OUTPUT_DIR / f"{xlsx_path.stem}_embeddings.npz"
        if output_path.exists():
            continue

        df = pd.read_excel(xlsx_path)
        titles = df["제목"].fillna("").astype(str).tolist()

        embeddings = model.encode(
            titles,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        np.savez(
            output_path,
            titles=np.array(titles, dtype=object),
            embeddings=embeddings.astype(np.float32),
        )

    print(f"done. outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
