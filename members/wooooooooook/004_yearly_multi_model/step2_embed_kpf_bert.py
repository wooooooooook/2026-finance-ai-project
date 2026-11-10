"""samples/sampled_pool.csv.gz를 kpf-bert-base-klueNLI-klueSTS(leewaay/kpf-bert-base-klueNLI-klueSTS-MSL512)로
임베딩한다. 한국언론진흥재단(KPF) 뉴스 코퍼스로 프리트레이닝된 뒤 KLUE-NLI/STS로
파인튜닝된 모델 - 뉴스 도메인과 STS 학습을 둘 다 만족하는 후보.

사용법:
    cd 004_yearly_multi_model
    uv run python step2_embed_kpf_bert.py
"""

from embed_common import run_embedding

MODEL_NAME = "leewaay/kpf-bert-base-klueNLI-klueSTS-MSL512"
KEY = "kpf_bert"

if __name__ == "__main__":
    run_embedding(KEY, MODEL_NAME)
