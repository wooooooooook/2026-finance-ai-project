"""samples/sampled_pool.csv.gz를 bge-m3-korean(upskyy/bge-m3-korean)으로 임베딩한다.
large급 모델(약 5.6억 파라미터)이라 base급 모델보다 4~5배 느리다 (실측 40.7건/초).

사용법:
    cd 004_yearly_multi_model
    uv run python step2_embed_bge_m3.py
"""

from embed_common import run_embedding

MODEL_NAME = "upskyy/bge-m3-korean"
KEY = "bge_m3"

if __name__ == "__main__":
    run_embedding(KEY, MODEL_NAME)
