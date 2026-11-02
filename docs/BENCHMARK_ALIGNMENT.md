# Benchmark 정합성 검토 및 수정

2026-11-02 서버의 `/home/work/Redteaming/Benchmark`를 기준으로 대조했다. 원래 실험의 추론·공격 생성·평가를 하나의 재사용 가능한 CLI로 정리하고, COVID-QA/TAT-QA와 명시적 target 설정은 확장으로 유지했다.

## 기준 코드

| 영역 | 원본 실행 경로 | ARS 구현 (`ars_bench/` 아래) |
|---|---|---|
| ReAct | `rag-exp/scripts/run_2.sh` → `ReAct/attack_react_so_kyuho.py` → `e5_env.py`/`wikienv.py` | `_benchmark/react.py`, `runners/react.py` |
| WebThinker | `Rag_uni/scripts/webthinker/4_TOTAL_PROCESS.sh` → deepseek/qwq attack-with-defense shell → `scripts/run_attack_defense.py` → `models/webthinker_defense.py` | `_benchmark/webthinker.py`, `runners/webthinker.py` |
| 공격 생성 | 양쪽 `generate_attack_doc_chanwoo_local_kill.py` | `attacks.py`, `_benchmark/attack_prompts.py`, `paradox_prompts.py`, `resources/` |
| 8개 평가 | `scripts/SUMMARY2.py` → `SUMMARY.py`; `run_IBS_GPT.py`, `run_ISS_GPT.py`, `run_PDS.py` | `benchmark_metrics.py`, `_benchmark/*`, `evaluation.py`, `judges.py` |
| 효율 2개 | `scripts/get_LI_and_TUI.py` | `_benchmark/efficiency.py`, `benchmark_metrics.py` |

원본 파일 및 선택한 함수/상수의 줄 번호·SHA256은 [sources.json](../ars_bench/_benchmark/sources.json)에 있다. shell 해시와 프롬프트 대조 결과는 [audit.json](benchmark_alignment_audit.json)에 있다. 절대경로·API key 초기화·모델 자동 실행을 복사하지 않고 알고리즘만 가져왔다. 외부 Benchmark runtime 의존성은 없다.

## 발견한 차이와 수정

| 항목 | 이전 ARS | 수정한 기본 프로토콜 |
|---|---|---|
| ReAct prompt | 다른 few-shot key | 원본 instruction + `webthink_subquery` |
| ReAct 제어 | 일반화한 parser/턴 | 9턴, 파싱 실패 시 추가 호출, 원본 stop/100토큰/온도0 |
| ReAct observation | 다른 문서 포맷 | title + newline + contents 결합 |
| ReAct 타깃 종료 | 일반 Finish 위주 | Action에 타깃이 포함되면 검색 이전에 전체 Action을 답으로 저장·종료 |
| WebThinker | main 검색 후 보조 요약 위주 | 실제 `process_single_sequence` + `generate_deep_web_explorer` |
| Web 모델 호출 | chat API 중심 | 원본 tokenizer template → completions → raw continuation |
| Web trace | 중첩 탐색 기록 부족 | main/explorer 검색, conclusion, 최종 thought와 explorer 원문 |
| Spotlight | 저장 passage 치환 | Web 원본 formatted observation/summary 경계 |
| PIGuard | 수동 logits/label 가정 | 원본 pipeline 및 injection/label_1/1 처리 |
| KidnapRAG | 추가 생성 지시/질문 prefix | 원본 messages/sampling, prefix 없이 bait5 + 체인25 |
| PoisonedRAG | 검색별 재생성 | 질문당5개 본문을 검색별 title로 재사용 |
| PARADOX | 검색당5개, 입력/온도 차이 | 검색당4개, subquery 원본3단계 prompt/온도1 |
| 첫 번째 평가 | Azure SBS | OpenAI moderation MBS |
| IBS/ISS | 0–5 rubric | 원본1–5 rubric/schema/입력 길이/토큰 한도 |
| IBS 범위 | 전체 | 기본 앞100개; `--document-limit -1`은 전체 평가 변형 |
| Telco utility | EM | 정규화 reference substring 포함 |
| Legal utility | 정규화 차이 | 원본 normalized/stemmed ROUGE-L, 빈 reference 제외 |
| UFS | clean=0 누락 | 원본대로 clean<=0이면0 |
| PDS | zero-transition 등 보정 | 원본 float32 DTW와 예외 동작 |
| LI | 질문별 main 지연 증가율 평균 | 총 wall time의 질문당 평균 간 비율 |
| TUI | tokenizer 실측/질문별 증가율 평균 | 원본 텍스트 추정식/질문당 평균 간 비율 |
| pipeline | 공통 count5/턴10 | ReAct9/Web10, PARADOX4 등 개별 기본값 |

KidnapRAG `build_messages`는 양쪽 원본 AST가 같다. PARADOX 3단계 prompt의 렌더링과 MAL-INS 5개도 양쪽과 일치한다. Encoder는 원본 `SimpleEncoder`의 줄 끝 공백을 정리했으며 실행 코드의 AST는 같다. 인덱스는 portability를 위해 manifest/document metadata가 있는 ARS 형식을 유지한다.

## 원본의 특이한 동작

- Deep Web Explorer는 QwQ일 때 main, 나머지는 aux 모델을 쓴다. chat template tokenizer 선택의 모델 이름 분기도 보존했다.
- E5 click은 실제 HTTP 페이지를 가져오지 않는다. 원본처럼 full document가 이미 있다는 문자열을 page-reader에 전달한다.
- search/click 중복 처리, interaction/단어 수 budget, answer extraction을 유지했다.
- 원본 ReAct step_stats에 search_query가 없으면 PDS용 legacy chain에 Action에서 추론해 넣지 않는다. ARS 검색 재사용용 normalized step에는 별도 보관한다.
- 원본 PDS는 정지 경로의 direction vector가0이면 두 경로가 같아도0이 아닐 수 있다. 같은 zero-transition 두 단계의 PDS=1.75를 유지했다.
- exposure는 원본처럼 파일 전체에서 search_documents/poisoned_flags schema를 선택한다. 이전 step별 혼합 해석은 instrumented에 남겼다.

## 유지한 확장과 재현 한계

1. COVID-QA/TAT-QA는 추가 데이터셋이며 normalized EM을 쓴다. TAT-QA 공식 전체 scorer와 같다고 주장하지 않는다.
2. target/intent를 명시하고 문서 ID를 유일하게 만든다. 원본의 고정 target, 중복 ID, stale metadata를 복제하지 않는다. 서버 호출, seed 전달, batch 방식은 이식 계층이므로 원본 vLLM 직접 batch와 bitwise 출력이 같다는 보장은 없다.
3. ReAct shell에는 방어 플래그가 있지만 지목된 `attack_react_so_kyuho.py`에는 해당 argparse/방어 연결이 없다. ARS 통합 방어는 실행 가능한 확장으로 유지한다. Web 방어는 원본 경계에 맞췄다. Task Shield 등 이번에 요청되지 않은 실험 방어는 CLI에 추가하지 않았다.
4. 데이터/기존 인덱스는 재샘플링·교체하지 않았다. 동일 수치를 비교하려면 질문 ID/순서, corpus, checkpoint/encoder revision, 생성 설정도 같아야 한다. 코퍼스/인덱스 불일치 시 조용히 truncate하는 대신 오류를 낸다. 소규모 개발 fixture의 top-k는 문서 수로 제한한다.
5. HTTP retry 계층과 원본 GPU in-process 호출은 다르다. 같은 temperature/seed라도 엔진·batching·GPU에 따라 출력이 달라질 수 있다.
6. 기존 runs/exports와 tar.gz는 재작성하지 않았다. 과거 clean의 프로토콜을 모르면 `baseline_protocol=legacy/unknown`으로 기록한다. 새 baseline은 `--clean`으로 지정할 수 있다.
7. 원본 시간 필드가 없는 결과는 LI를 복구할 수 없다. main latency를 총 wall time으로 대체하지 않는다. 새 Web 실행은 tokenizer 준비 이후 batch 추론을 측정한다. 동시 요청 한도/하드웨어를 같게 유지해야 LI 비교가 의미 있다.
8. 이전 추론은 `--protocol simplified`, 평가는 `--metric-profile instrumented`에 남겼다. CLI 기본은 Benchmark이며 cache/CSV profile을 분리한다. Python `evaluation.evaluate()`는 기존 호출 호환성 때문에 생략 시 instrumented를 유지한다. 라이브러리에서는 `profile="benchmark"`를 명시한다.

## 검증과 보존

- 모델/API 없는 60개 테스트로 ReAct fallback/타깃 종료/9턴, Web 중첩 검색·클릭·continuation/main-aux 호출 수, sampling/문서 생성 단위, utility/PDS/LI/TUI, raw trace/cache/HTTP workflow를 검증했다.
- 복사 함수/상수 해시를 고정해 우발적 알고리즘 변경을 감지한다.
- 실제 GPU LLM 추론과 유료 moderation/judge 호출은 수행하지 않았다. 전체 데이터셋 ASR/utility 수치 재현 검증은 별도다.
- 수정 전 소스는 `.cache/before_benchmark_alignment_20261102_163906/`에 백업했다. 기존 runs/문서 패키지/모델/인덱스는 보존했다.
