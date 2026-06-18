# EV-QA-Framework Roadmap

**Version:** 2.1.0 → Target 2.2.0+  
**Date:** 2026-06-18  
**Based on:** Code audit (code_quality_audit.md), STATUS.md, pyproject.toml, CI/CD configs

---

## Current State

### ✅ What Works
- **22 modules** in `ev_qa_framework/` — all properly exported via `__init__.py` (205 entries)
- **901 tests passing** (updated from 592 in PROJECT_STRUCTURE.md)
- **87% coverage** overall; 100% on `__init__.py`, `v2g_scenarios.py`; 98%+ on `chemistries.py`, `soh_predictor.py`, `physics_features.py`
- **No security issues** — no secrets, SQL injection, eval/exec, unsafe deserialization, path traversal
- **No Cyrillic** in source code (`ev_qa_framework/`, `tests/`)
- **Ruff linting**: only 2 issues (1 auto-fixable import sort in `bms_protocol.py`; 1 notebook cell in `anomaly_detection_demo.ipynb`)
- **CI/CD**: GitHub Actions — lint (ruff) → test matrix (3.10/3.11/3.12) → coverage (Codecov) → release (trusted PyPI publish)
- **Docker**: Single-stage `python:3.12-slim` + uv, healthcheck, entrypoint `ev-qa`
- **docker-compose**: 3 services (tests → coverage dashboard → Grafana)
- **Pre-commit**: ruff (fix+format), trailing-whitespace, end-of-file-fixer, check-yaml/json, merge-conflict, debug-statements, check-ast, pytest (pre-push)
- **Sphinx docs**: GitHub Pages deployment configured (pages.yml)
- **PyPI trusted publishing** workflow configured (release.yml)

### ❌ What's Broken / Needs Work
| Area | Issue | Evidence |
|------|-------|----------|
| **Python version alignment** | `pyproject.toml` says `>=3.10`, classifiers list 3.9–3.12, Dockerfile uses 3.12, CI tests 3.10/3.11/3.12, README badge says 3.9+ | Inconsistent |
| **Ruff config** | `target-version = "py39"` but `requires-python = ">=3.10"`; `N806` ignored globally but used in async methods | Misaligned |
| **Coverage restoration** | 4 modules <85%: `can_bus.py` 73%, `analysis.py` 73%, `thermal_runaway.py` 75%, `hil.py` 81% | Audit report |
| **Docstrings** | 73 missing — mostly `__init__`, context managers, connection methods, serialization in `chemistries.py` | Audit report |
| **Test thresholds** | 7 test files updated locally but NOT committed (STATUS.md) | STATUS.md |
| **modbus.py** | `_extract_pdu` missing TID validation; needs `self._last_tid` vs MBAP header check | STATUS.md |
| **can_bus.py** | Bus-off in receive path, DLC validation, J1939 29-bit ID validation incomplete | STATUS.md |
| **Docker healthcheck** | Checks version `2.0.0` but package is `2.1.0` | Dockerfile line 27 |
| **Release version gate** | Workflow extracts version from tag but verifies against `ev_qa_framework.__version__` — no `__version__` export visible | release.yml line 84 |

---

## Immediate (1-2 weeks)

1. **Commit threshold-fixed tests**
   - `git commit --no-verify -m "fix: update test thresholds to match production defaults"` (7 files per STATUS.md)
   - Run full test suite in batches: `pytest tests/test_X.py tests/test_Y.py -q --tb=line`

2. **Fix modbus.py TID validation**
   - Add `_last_tid` tracking vs MBAP header bytes[0:2] in `_extract_pdu`

3. **Fix can_bus.py critical paths**
   - Bus-off handling in receive path
   - DLC validation (0–8 bytes)
   - J1939 29-bit ID range validation (already noted in CHANGELOG v2.0.0 fixed but verify)

4. **Align Python version metadata**
   - `pyproject.toml`: set `requires-python = ">=3.10"` (keep), update classifiers to 3.10/3.11/3.12 only (drop 3.9)
   - Dockerfile: keep `python:3.12-slim` or update to 3.12-bookworm
   - CI matrix: 3.10, 3.11, 3.12 (already correct)
   - README badge: change `3.9+` → `3.10+`

5. **Fix Ruff config alignment**
   - `tool.ruff.target-version = "py310"` (match requires-python)
   - Remove `N806` from global ignore (used for async method first arg); add per-file if needed

6. **Run ruff auto-fix**
   - `uv run ruff check --fix .` (fixes bms_protocol.py import sort)

7. **Fix Docker healthcheck version**
   - Update `assert ev_qa_framework.__version__ == '2.1.0'` in Dockerfile line 27

8. **Add `__version__` to package**
   - Export `__version__ = "2.1.0"` in `ev_qa_framework/__init__.py` for release workflow verification

---

## Short-term (1 month)

1. **Coverage restoration — target >85% on all modules**
   - `can_bus.py` (73% → 85%): hardware interface paths, error handling, CANHardwareInterface, OBD2Adapter
   - `analysis.py` (73% → 85%): ML anomaly detection branches, edge cases in EVBatteryAnalyzer
   - `thermal_runaway.py` (75% → 85%): prediction edge cases, single-row DataFrame handling
   - `hil.py` (81% → 85%): HIL test runner paths, BMSHardwareEmulator

2. **Docstring completion — public APIs first**
   - `chemistries.py`: 12 serialization methods (`to_dict`, `from_dict`, `to_json`, `from_json`, `save_to_file`, `load_from_file`)
   - `can_bus.py`, `bms_protocol.py`, `modbus.py`, `hil.py`: connection management (`__init__`, `is_connected`, context managers)
   - Remaining `__init__` methods (42 total) — prioritize public classes

3. **Fix pre-commit hooks**
   - Current pytest hook runs full suite on pre-push (slow); consider subset or move to CI-only
   - Verify ruff-pre-commit rev `v0.5.0` is current (latest is ~v0.5.x)

4. **Test suite unification**
   - Current: 38 test files, some overlap (e.g., `test_ev_qa_anomalies.py` vs `test_advanced_analysis.py`)
   - Consolidate integration tests: `test_integration.py` + `test_integration_extended.py` → single `test_integration/`
   - Separate unit vs integration via `pytest` markers (`@pytest.mark.integration`)

5. **Docker cleanup**
   - Multi-stage build (builder + runtime) per CHANGELOG v2.0.0 claim — currently single-stage
   - Pin uv version in Dockerfile (currently `ghcr.io/astral-sh/uv:0.5` — update to latest)
   - Add `.dockerignore` optimization (exclude `.venv`, `__pycache__`, `.pytest_cache`, `docs/build`, `.ruff_cache`)

---

## Medium-term (3 months)

1. **PyPI publish — first public release**
   - Tag `v2.2.0` after Immediate + Short-term items done
   - Verify trusted publishing workflow (release.yml) works end-to-end
   - Ensure `CHANGELOG.md` updated with 2.2.0 entries
   - Test install: `pip install ev-qa-framework==2.2.0`

2. **Docs generation — Sphinx API reference**
   - Verify `docs/source/conf.py` exists and builds (pages.yml assumes it does)
   - Run `uv run sphinx-build -b html docs/source docs/build/html` locally
   - Enable GitHub Pages deployment (already in pages.yml)
   - Add docstring coverage badge to README

3. **Architecture cleanup**
   - God-object review: `framework.py` (`EVQAFramework` facade), `bms_protocol.py` (380 lines), `can_bus.py` (659 lines)
   - Extract protocols: `bms_protocol.py` → `can_interface.py`, `modbus_interface.py`
   - Extract CAN hardware: `can_bus.py` → `can_hardware.py`, `can_simulation.py`, `dbc_integration.py`
   - Introduce interface segregation: `BMSProtocol` abstract base class

4. **Dependency modernization**
   - `requirements.txt` pinned but `uv.lock` is source of truth — consider removing requirements.txt
   - Review optional deps: `ml` (tensorflow 2.15), `hardware` (python-can, pyserial) — versions in pyproject.toml not pinned
   - Add `dependency-groups` for `test`, `lint`, `docs` (already has `dev`, `ml`, `hardware`, `docs`)

5. **HIL & hardware testing**
   - Add real BMS telemetry adapters (Tesla, BYD, Nio) — per README roadmap
   - CI pipeline for hardware-in-loop (needs runner with CAN interface)

---

## Long-term (6+ months)

1. **V2G + V2S scenarios expansion**
   - Charging-station integration scenarios
   - Grid services: frequency regulation, peak shaving, solar buffering (already in v2g_scenarios.py)

2. **Real BMS adapters**
   - Tesla BMS CAN protocol reverse-engineering
   - BYD/Nio protocol adapters
   - OBD-II ELM327 live data streaming

3. **Vector CANoe/CANalyzer integration**
   - Export test vectors for commercial toolchains
   - DBC round-trip validation

4. **AutoML production hardening**
   - Optuna integration for HPO (currently optional)
   - Model registry + versioning for SOH/anomaly models
   - ONNX export for edge deployment

5. **Fleet analytics at scale**
   - Parquet/Arrow support for large datasets
   - Distributed processing (Dask/Ray)
   - Real-time streaming (Kafka/Redpanda)

6. **Compliance certification**
   - Formal UN 38.3 / IEC 62660 / GB 38031 test report generation
   - Audit trail for automotive functional safety (ISO 26262)

---

## Quick Reference: Priority Matrix

| Priority | Task | Effort | Blocked By |
|----------|------|--------|------------|
| 🔴 P0 | Commit threshold tests | 1h | — |
| 🔴 P0 | Fix modbus TID validation | 2h | — |
| 🔴 P0 | Fix can_bus critical paths | 4h | — |
| 🔴 P0 | Align Python version metadata | 1h | — |
| 🔴 P0 | Fix Ruff target-version | 15m | — |
| 🔴 P0 | Run ruff --fix | 1m | — |
| 🔴 P0 | Fix Docker healthcheck version | 5m | — |
| 🔴 P0 | Add __version__ export | 10m | — |
| 🟡 P1 | Coverage >85% on 4 modules | 2-3 days | P0 done |
| 🟡 P1 | Docstrings on public APIs | 1-2 days | — |
| 🟡 P1 | Fix pre-commit pytest hook | 30m | — |
| 🟡 P1 | Test suite unification (markers) | 1 day | — |
| 🟡 P1 | Docker multi-stage build | 2h | — |
| 🟢 P2 | PyPI publish v2.2.0 | 2h | P1 done |
| 🟢 P2 | Sphinx docs + GitHub Pages | 2h | — |
| 🟢 P2 | Architecture cleanup (god objects) | 1-2 weeks | — |
| 🔵 P3 | Real BMS adapters | 1-2 months | — |
| 🔵 P3 | Vector CANoe integration | 1 month | — |
| 🔵 P3 | AutoML + model registry | 2 months | — |