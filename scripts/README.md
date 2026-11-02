# Scripts

표준 실행은 레포 루트의 `./ars index`, `run`, `generate`, `evaluate`, `aggregate`, `pipeline`이다. 절차는 `README.md`, 원본 대조는 `docs/BENCHMARK_ALIGNMENT.md`에 있다.

- `SUMMARY2.py`: 새 bundle evaluator의 호환 진입점. 기본 Benchmark 10지표. 원본의 파일명 탐색 인자는 사용하지 않는다.
- `get_LI_and_TUI.py`: bundle의 Benchmark LI/TUI. `--metric-profile instrumented`로 이전 정의 선택 가능.
- `prepare_qa_datasets.py`, `import_benchmark_kidnaprag.py`: 데이터 준비/기존 자산 이관.
- `make_demo.py`: 명시적인 synthetic fixture.
- `kidnaprag_*`, `generate_kidnaprag_bait.py`, `evaluate_kidnaprag_*`, `matrix_e5_service.py`: 기존 대량 실험/분석 도구. 배포 run/package 참조를 보존하기 위해 경로를 유지했다. 과거 결과를 새 Benchmark 프로토콜 결과로 재분류하지 않는다.

새 실험은 `./ars`를 사용한다. 기존 도구를 재사용할 때는 명시적인 설정과 원본 경로를 확인한다.
