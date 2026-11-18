"""
cluster_explore.py: 010 실험 — 001~004에서 쓴 HDBSCAN(min_cluster_size=15, 2차원 UMAP/t-SNE
좌표 위에서 실행)은 클러스터가 수백 개까지 쪼개지는 문제가 있었다(004/step3_cluster.py 참고).
이 실험은 "클러스터 개수를 직접 통제"하는 걸 목표로, 임베딩 모델 3개(sbert/bge_m3/kpf_bert,
004와 동일) x 축소 차원(4/8/16) x 클러스터링 방법(KMeans 등 — k 직접 지정 / HDBSCAN —
min_cluster_size로 간접 통제)을 조합해서 돌려보고, 각 조합의 클러스터 개수·품질 지표·2차원
산점도를 남긴다. 여기서 "쓸 만한 조합"(모델/차원/방법/파라미터)을 고르면, 그 설정을 011·012
같은 다음 실험에서 그대로 쓴다. 뉴스 자체를 수익률로 예측하는 단계는 없다 — 클러스터링 방법론
선택용 탐색 실험.

비자명한 결정:
- AgglomerativeClustering은 뺐다. 샘플 수(월 300개 x 300개월 = 9만 건 근처)에서 ward
  연결은 O(n^2) 거리행렬이 필요해 메모리가 감당 안 됨.
- HDBSCAN은 "2차원으로 줄인 좌표가 아니라, 4/8/16차원으로 줄인 좌표에서" 실행한다
  (004는 2차원 UMAP 좌표에서 돌려서 클러스터가 지나치게 쪼개졌을 가능성이 있음 — 고차원일수록
  점들이 덜 붙어있어 큰 클러스터가 나오는지 비교해보려는 목적도 있음).
- 시각화용 2차원 UMAP은 클러스터링 입력 차원과 분리해서 모델당 1번만 계산하고, 모든 조합의
  산점도에 재사용한다(클러스터링은 4/8/16차원에서, 그림은 항상 2차원 좌표 위에 색칠).
- 임베딩은 004와 동일하게 sentence-transformers로 통일(SBERT/BGE-M3/KPF-BERT 셋 다
  SentenceTransformer로 로드 가능 — 모델마다 다른 pooling 코드를 안 짜도 됨).

실행 순서:
  1. 뉴스 월별 샘플링(008/009와 동일한 일자가중 방식). samples/pool.csv.gz 캐시
  2. 임베딩 모델 3개로 각각 임베딩. embeddings/{key}.npz 캐시
  3. 모델별로 2차원 UMAP(시각화용) 1회 + 4/8/16차원 UMAP(클러스터링 입력용) 계산
  4. 각 (모델, 차원, 방법, 파라미터) 조합에 대해 KMeans(k=8/16/32) 또는
     HDBSCAN(min_cluster_size=100/300) 클러스터링 → 품질 지표 + 2차원 산점도
  5. 전체 조합 요약 테이블(클러스터 개수 vs silhouette) 이미지

사용법:
    cd members/wooooooooook/010_news_cluster_explore
    python3 cluster_explore.py

    # CPU 코어 일부만 쓰고 싶을 때 (기본 -1 = 전체 코어)
    N_JOBS=6 python3 cluster_explore.py
"""

import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
RAW = PROJECT_ROOT / "raw"

SAMPLE_PATH = BASE_DIR / "samples" / "pool.csv.gz"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
CLUSTERS_DIR = BASE_DIR / "clusters"
IMAGES_DIR = BASE_DIR / "images"

SEED = 42
N_CORES = os.cpu_count() or 1
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

# 뉴스 샘플링 (008/009와 동일한 일자가중 방식, 규모만 줄임 — 방법론 탐색용이라
# 전체 기간을 다 쓰되 월 300개로 캡)
NEWS_PER_MONTH = 300
NEWS_PER_DAY_CAP = 300
NEWS_BODY_CHARS = 200

EMBED_MODELS = {
    "sbert": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    "bge_m3": "upskyy/bge-m3-korean",
    "kpf_bert": "leewaay/kpf-bert-base-klueNLI-klueSTS-MSL512",
}

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
REDUCE_DIMS = [4, 8, 16]          # 클러스터링 입력 차원
PLOT_DIM = 2                       # 시각화용 차원 (고정)

K_VALUES = [8, 16, 32]             # KMeans: 직접 지정
HDBSCAN_MIN_SIZES = [100, 300]     # HDBSCAN: min_cluster_size로 간접 통제


# ===========================================================================
# 1. 뉴스 월별 샘플링 (008/009와 동일 로직)
# ===========================================================================

def _read_news_day(path_str: str):
    path = Path(path_str)
    month = path.stem.replace("bigkinds_", "")[:7]
    d = pd.read_csv(path, usecols=["제목", "본문", "분석제외 여부"], dtype=str,
                    keep_default_na=False, encoding="utf-8-sig")
    d = d[(d["제목"].str.strip() != "") & (d["본문"].str.strip() != "")]
    d = d[d["분석제외 여부"].str.strip() == ""]
    d = d.drop_duplicates(["제목", "본문"])
    vc = d["제목"].value_counts()
    bad = vc[(vc >= 10) & (vc >= 0.01 * len(d))].index
    d = d[~d["제목"].isin(bad)]
    n = len(d)
    if n > NEWS_PER_DAY_CAP:
        d = d.sample(NEWS_PER_DAY_CAP, random_state=zlib.crc32(path.name.encode()) % (2 ** 32))
    text = (d["제목"].str.strip() + ". "
            + d["본문"].str.slice(0, NEWS_BODY_CHARS).str.replace(r"\s+", " ", regex=True))
    return month, n, text.tolist()


def _sample_month(days: list, rng) -> list[str]:
    texts, weights = [], []
    for n, t in days:
        if t:
            texts.extend(t)
            weights.extend([n / len(t)] * len(t))
    if not texts:
        return []
    p = np.asarray(weights) / np.sum(weights)
    idx = rng.choice(len(texts), size=min(NEWS_PER_MONTH, len(texts)), replace=False, p=p)
    return [texts[i] for i in sorted(idx)]


def stage_sample() -> None:
    print("\n" + "=" * 70 + "\n[1] 뉴스 월별 샘플링\n" + "=" * 70)
    if SAMPLE_PATH.exists():
        print(f"이미 존재 -> {SAMPLE_PATH}")
        return

    files = sorted(str(p) for p in (RAW / "news").glob("bigkinds_*.csv.gz"))
    print(f"일별 파일 {len(files)}개")
    rng = np.random.default_rng(SEED)
    rows = []
    cur_month, days = None, []

    def flush():
        if cur_month is None:
            return
        for t in _sample_month(days, rng):
            rows.append((cur_month, t))

    with ProcessPoolExecutor(max_workers=N_CORES) as pool:
        for month, n, texts in tqdm(pool.map(_read_news_day, files, chunksize=8),
                                    total=len(files), desc="news files"):
            if month != cur_month:
                flush()
                cur_month, days = month, []
            days.append((n, texts))
        flush()

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["month", "text"]).to_csv(SAMPLE_PATH, index=False)
    print(f"저장 완료 -> {SAMPLE_PATH}, {len(rows)}건")


# ===========================================================================
# 2. 임베딩 (004와 동일 — sentence-transformers로 3개 모델 통일)
# ===========================================================================

def stage_embed(key: str, model_name: str) -> None:
    out_path = EMBEDDINGS_DIR / f"{key}.npz"
    if out_path.exists():
        print(f"[{key}] 이미 존재 -> {out_path}")
        return

    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(SAMPLE_PATH, dtype={"month": str})
    texts = df["text"].fillna("").astype(str).tolist()
    print(f"[{key}] 임베딩 대상 {len(texts)}건, model={model_name}")

    model = SentenceTransformer(model_name, device="cpu")
    t0 = time.time()
    emb = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    print(f"[{key}] {time.time() - t0:.0f}s, {len(texts) / (time.time() - t0):.1f}건/초")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, month=df["month"].to_numpy().astype(str), embeddings=emb.astype(np.float32))
    print(f"[{key}] 저장 완료 -> {out_path}, dim={emb.shape[1]}")


# ===========================================================================
# 3~4. 차원 축소 + 클러스터링 + 품질 지표
# ===========================================================================

def _metrics(raw_emb: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2 or mask.sum() < 2:
        return {"n_clusters": n_clusters, "noise_ratio": float((~mask).mean()),
                "silhouette": np.nan, "calinski": np.nan, "davies": np.nan}
    sub = raw_emb[mask]
    lab = labels[mask]
    # 대규모 샘플이면 silhouette 계산이 느려서 일부만 사용
    if len(sub) > 20000:
        idx = np.random.default_rng(SEED).choice(len(sub), 20000, replace=False)
        sub, lab = sub[idx], lab[idx]
    return {
        "n_clusters": n_clusters,
        "noise_ratio": float((~mask).mean()),
        "silhouette": float(silhouette_score(sub, lab)),
        "calinski": float(calinski_harabasz_score(sub, lab)),
        "davies": float(davies_bouldin_score(sub, lab)),
    }


def _plot(coords2d: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    rng = np.random.default_rng(0)
    n = len(labels)
    max_points = 20000
    idx = rng.choice(n, max_points, replace=False) if n > max_points else np.arange(n)
    idx = rng.permutation(idx)
    c, lab = coords2d[idx], labels[idx]

    unique_vals = sorted(v for v in set(lab) if v != -1)
    cmap = plt.get_cmap("tab20")
    color_map = {v: cmap(i % 20) for i, v in enumerate(unique_vals)}
    color_map[-1] = (0.85, 0.85, 0.85, 1.0)
    colors = [color_map[v] for v in lab]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.scatter(c[:, 0], c[:, 1], s=6, c=colors, alpha=0.4, linewidths=0)
    ax.set_title(f"{title} (plotted {len(idx)}/{n})")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def process_model(key: str) -> list[dict]:
    header = f"[{key}] 차원 축소 + 클러스터링"
    print("\n" + "=" * 70 + f"\n{header}\n" + "=" * 70)
    npz = np.load(EMBEDDINGS_DIR / f"{key}.npz", allow_pickle=True)
    raw_emb = npz["embeddings"]
    print(f"[{key}] 로드: {len(raw_emb)}건, 원본 dim={raw_emb.shape[1]}")

    print(f"[{key}] 시각화용 {PLOT_DIM}차원 UMAP...")
    coords2d = umap.UMAP(n_components=PLOT_DIM, n_neighbors=UMAP_N_NEIGHBORS,
                        min_dist=UMAP_MIN_DIST, random_state=SEED, n_jobs=N_JOBS).fit_transform(raw_emb)

    records = []
    for dim in REDUCE_DIMS:
        print(f"[{key}] 클러스터링용 {dim}차원 UMAP...")
        reduced = umap.UMAP(n_components=dim, n_neighbors=UMAP_N_NEIGHBORS,
                            min_dist=UMAP_MIN_DIST, random_state=SEED, n_jobs=N_JOBS).fit_transform(raw_emb)

        specs = [("kmeans", k, KMeans(n_clusters=k, random_state=SEED, n_init=10)) for k in K_VALUES]
        specs += [("hdbscan", mcs, HDBSCAN(min_cluster_size=mcs, n_jobs=N_JOBS)) for mcs in HDBSCAN_MIN_SIZES]

        for algo, param, clusterer in specs:
            t0 = time.time()
            labels = clusterer.fit_predict(reduced)
            m = _metrics(reduced, labels)
            m.update({"model": key, "dim": dim, "algo": algo, "param": param, "fit_s": time.time() - t0})
            records.append(m)
            print(f"  [{key}/dim{dim}/{algo}={param}] n_clusters={m['n_clusters']}, "
                  f"noise={m['noise_ratio']:.1%}, silhouette={m['silhouette']:.3f} ({m['fit_s']:.1f}s)")

            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            _plot(coords2d, labels,
                 f"{key} dim={dim} {algo}={param} (n_clusters={m['n_clusters']})",
                 IMAGES_DIR / f"{key}_dim{dim}_{algo}{param}.png")

    return records


# ===========================================================================
# 5. 요약 이미지 (클러스터 개수 vs silhouette)
# ===========================================================================

def stage_report(all_records: list[dict]) -> None:
    df = pd.DataFrame(all_records)
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLUSTERS_DIR / "sweep_metrics.csv", index=False)

    fig, axes = plt.subplots(1, len(EMBED_MODELS), figsize=(5.5 * len(EMBED_MODELS), 5), sharey=True)
    markers = {"kmeans": "o", "hdbscan": "^"}
    for ax, key in zip(axes, EMBED_MODELS):
        sub = df[df["model"] == key]
        for algo, g in sub.groupby("algo"):
            for dim, gg in g.groupby("dim"):
                ax.scatter(gg["n_clusters"], gg["silhouette"], label=f"{algo} dim{dim}",
                          marker=markers.get(algo, "x"), s=50)
        ax.set_title(key)
        ax.set_xlabel("n_clusters")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("silhouette (clustering-input space)")
    fig.suptitle("클러스터 개수 vs silhouette — 조합별 비교 (적은 클러스터 + 높은 silhouette가 이상적)")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "sweep_summary.png", dpi=150)
    plt.close(fig)
    print(f"\n요약 저장 완료 -> {CLUSTERS_DIR}/sweep_metrics.csv, {IMAGES_DIR}/sweep_summary.png")
    print("\n" + df.sort_values(["model", "dim", "algo"]).to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    stage_sample()
    for key, model_name in EMBED_MODELS.items():
        stage_embed(key, model_name)

    all_records = []
    for key in EMBED_MODELS:
        all_records += process_model(key)
    stage_report(all_records)


if __name__ == "__main__":
    main()
