"""samples/sampled_pool.csv.gz를 KR-SBERT(snunlp/KR-SBERT-V40K-klueNLI-augSTS)로 임베딩한다.

사용법:
    cd 004_yearly_multi_model
    uv run python step2_embed_sbert.py
"""

from embed_common import run_embedding

MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
KEY = "sbert"

if __name__ == "__main__":
    run_embedding(KEY, MODEL_NAME)
