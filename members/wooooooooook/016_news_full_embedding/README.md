# 016_news_full_embedding

014(news_horizon_decay)는 비용 때문에 뉴스를 10% 무작위 샘플링해서 임베딩했다. 이 실험은
**전체(100%) 뉴스**를 샘플링 없이 다 필터링+임베딩해서 캐싱해본다 — 노트북에 디스크를
200GB로 늘려서 용량 문제(기존에 서버 디스크 33GB로는 전체 임베딩 시 필요한 ~90GB가 안
들어갔음)를 해결했기 때문에 시도 가능해짐.

014와 분리한 이유: 다른 실험 폴더끼리 import 금지 원칙 + 이게 성공/실패하든 014의 10%
샘플 결과(이미 거의 다 돼 있음)에 영향을 주지 않게 하기 위함.

## 설계

- **필터링은 014와 동일**(PREPROCESSING.md 1~4·6번), **샘플링만 뺌**(100% 사용)
- **raw/news에서 바로 필터링+임베딩**: 014처럼 중간에 전처리된 텍스트 파일을 따로 안
  만듦 — raw/news 원본이 이 스크립트를 돌리는 기기(서버·노트북 둘 다)에 이미 있어서,
  중간 파일을 만들고 옮기는 단계를 생략함
- **날짜 단위 캐싱**(`news_cache/emb_{YYYY-MM-DD}.npz`, 하루 1파일): 연 단위(014)보다
  훨씬 촘촘한 재개 단위. 전체 작업이 2~3일(노트북 MPS 기준 추정) 걸리는 걸 감안해 중단
  시 손실을 최소화
- **cuda/mps/cpu 자동 감지**: 어느 기기에서 돌려도 됨(014에 추가한 것과 동일한 `_device()`)
- **메모리 정리를 날짜마다**: 014에서 겪은 메모리 누수(연 단위 반복 시 RSS가 62.8GB까지
  차서 프로세스 강제종료된 사건) 재발 방지 — `gc.collect()`+`malloc_trim(0)`, MPS면
  `torch.mps.empty_cache()`도 추가
- **교차검증용 카운트 기록**: `news_cache/counts.csv`(날짜별 n_filtered)를 계속 누적
  저장하고, 끝나면 합계가 014에서 이미 확인한 전체 필터 통과 기사 수(29,356,582)와
  일치하는지 확인

## 비자명한 부분

- **배치 크기를 고정하지 않고 시작할 때 자동으로 고름**(`_calibrate_batch_size`):
  32/64/128/256 중 기사 수가 가장 많은 날 샘플로 각각 속도를 재서(워밍업 1회+3회 반복)
  가장 빠른 걸 전체 작업에 씀. 추측으로 정하지 않고 실측해서 고르는 이유는, GPU/MPS
  배치 최적값이 기기·모델·시퀀스 길이에 따라 달라 미리 알 수 없기 때문.
- npz 파일 하나에 날짜가 하나뿐이라(014는 연 단위라 여러 날짜가 섞여 있어서 `date`
  배열이 따로 필요했음), 이 버전은 `embeddings`만 저장함 — 파일명 자체가 날짜임.
- 저장 위치는 `raw/`가 아니라 이 폴더 `news_cache/`(가공본이라 raw 폴더 원칙과 안 맞음).

## 실행법

```bash
# 노트북(맥)에서
caffeinate -i uv run python members/wooooooooook/016_news_full_embedding/news_full_embedding.py 2>&1 | tee -a members/wooooooooook/016_news_full_embedding/logs/run.log

# 서버에서
uv run python members/wooooooooook/016_news_full_embedding/news_full_embedding.py | tee -a members/wooooooooook/016_news_full_embedding/logs/run.log
```

tmux/screen으로 세션 유지 필수(2~3일 걸리는 작업). 날짜 단위로 캐싱되므로 중단 후
재실행하면 끝난 날짜는 건너뜀.

## 결과

아직 실행 안 함.
