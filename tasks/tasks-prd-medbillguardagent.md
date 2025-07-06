# Task List: MedBillGuardAgent Implementation (CTO-validated)

Below is the **final task list** that aligns 1-for-1 with the PRD, Engineering Ruleset, and Architecture Layers.

---

## Relevant Files

### Core Service Files
| File | Purpose |
|------|---------|
| `medbillguardagent.py` | FastAPI service: endpoints, async business flow - COMPLETE |
| `test_medbillguardagent.py` | Unit & integration tests (≥90 % coverage) |
| `test_api.py` | Simple API endpoint testing script - COMPLETE |
| `schemas.py` | Pydantic – `MedBillGuardRequest`, `MedBillGuardResponse`, `RedFlag` |
| `test_schemas.py` | Schema validation tests |

### Business Logic & Processing
| File | Purpose |
|------|---------|
| `medbillguardagent/document_processor.py` | OCR → table extraction - COMPLETE |
| `tests/test_document_processor.py` | Tests for processor - COMPLETE |
| `medbillguardagent/duplicate_detector.py` | Duplicate medical test/procedure detection - COMPLETE |
| `tests/test_duplicate_detector.py` | Tests for duplicate detection - COMPLETE |
| `medbillguardagent/prohibited_detector.py` | Prohibited item detection via data/prohibited.json - COMPLETE |
| `tests/test_prohibited_detector.py` | Tests for prohibited detection - COMPLETE |
| `rate_validator.py` | CGHS/ESI/NPPA comparison, over-charge calc - COMPLETE |
| `test_rate_validator.py` | Validation tests - COMPLETE |
| `confidence_scorer.py` | Rule + LLM hybrid confidence - COMPLETE |
| `test_confidence_scorer.py` | Confidence scorer tests - COMPLETE |
| `cache_manager.py` | Redis-backed caching with 24h TTL - COMPLETE |
| `test_cache_manager.py` | Cache manager tests - COMPLETE |
| `explanation_builder.py` | Markdown + SSML generator |
| `test_explanation_builder.py` | Tests for explanations |

### Data & Reference Management
| File | Purpose |
|------|---------|
| `reference_data_loader.py` | Load & cache CGHS, ESI, NPPA data - COMPLETE |
| `nightly_job.py` | Cron / Celery beat for nightly refresh |
| `test_reference_data_loader.py` | Data loader tests - COMPLETE |
| `data/cghs_rates_2023.json` | Processed CGHS tariff - COMPLETE |
| `data/esi_rates.json` | ESI rates - COMPLETE |
| `data/nppa_mrp.json` | NPPA drug MRPs - COMPLETE |
| `data/nppa_mrp.csv` | NPPA drug MRPs (CSV format) - COMPLETE |
| `data/state_tariffs/` | State-specific rates - COMPLETE |
| `data/state_tariffs/delhi.json` | Delhi state tariffs - COMPLETE |
| `data/state_tariffs/maharashtra.json` | Maharashtra state tariffs - COMPLETE |
| `data/state_tariffs/karnataka.json` | Karnataka state tariffs - COMPLETE |
| `data/prohibited.json` | Prohibited fee list - COMPLETE |

### Shared Libraries (already exist)
| File | Purpose |
|------|---------|
| `libs/ocr.py` | PyMuPDF → PNG → multi-lang Tesseract |
| `libs/llm_provider.py` | LLM interface (OpenAI / local) |
| `libs/vector_search.py` | LanceDB / Weaviate wrapper |
| `libs/testing.py` | `@timed_test`, snapshot helper - COMPLETE |
| `libs/feature_flags.py` | Unleash client helper |

### Configuration & Infrastructure
| File | Purpose |
|------|---------|
| `pyproject.toml` | Poetry project config, dependencies, dev tools (Ruff, MyPy, Pytest) |
| `Dockerfile` | Multi-stage build: builder + runtime with OCR dependencies |
| `.dockerignore` | Optimized build context exclusions |
| `docker-compose.yml` | Local dev stack: Redis, NATS, MinIO, LanceDB, Unleash + monitoring |
| `monitoring/prometheus.yml` | Prometheus scrape config for all services |
| `.github/workflows/ci.yml` | 7-stage CI/CD: lint, typecheck, security, test, perf, docker, deploy |
| `locustfile.py` | Load testing with Locust for performance validation |
| `config/default.yaml`, `config/production.yaml` | Config |
| `k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/configmap.yaml`, `k8s/hpa.yaml` | K8s manifests |
| `monitoring/prometheus.yml` | Metric scrape |
| `otel-collector-config.yaml` | (optional) OTEL collector |

### Prompts & Templates
| File | Purpose |
|------|---------|
| `prompts/medbillguard.j2` | Jinja2 prompt template |
| `prompts/medbillguardagent.md` | Cursor prompt (committed) |
| `templates/refund_letter.html` / `.docx` | Action templates |

### Testing & Fixtures
| File | Purpose |
|------|---------|
| `fixtures/example.pdf` | Golden fixture |
| `fixtures/expected_output.json` | Snapshot JSON |
| `fixtures/cghs_sample_bill.pdf` | Sample bill |
| `fixtures/pharmacy_invoice.pdf` | Sample pharmacy invoice |
| `loadtest/locustfile.py` | Locust perf test |

### Monitoring & Deployment
| File | Purpose |
|------|---------|
| `SECURITY.md` | DPDP & DevSec guidelines |
| `README.md`, `API.md`, `DEPLOYMENT.md` | Docs |

### New Testing Library and Performance Test Files
| File | Purpose |
|------|---------|
| `tests/test_performance.py` | Performance tests with @timed_test decorator - COMPLETE |

---

## Tasks

### 1. Project Setup & Infrastructure
- [x] **1.1** Initialise Poetry project (`pyproject.toml`) + dependencies  
- [x] **1.2** Build multi-stage **Dockerfile** (python:3.11-slim, poetry install)  
- [x] **1.3** `docker-compose.yml` with Redis, NATS, MinIO, LanceDB, Unleash mock  
- [x] **1.4** CI pipeline (`ci.yml`): Ruff → MyPy → Pytest → Coverage → Trivy → Docker build  
- [x] **1.5** K8s manifests (`deployment`, `service`, `configmap`, `hpa`)  
- [x] **1.6** Config management (`config/*.yaml` + ENV overrides)  
- [x] **1.7** Directory skeleton + pre-commit hooks

### 2. Document Processing & OCR Pipeline
- [x] **2.1** Create `schemas.py` + tests  
- [x] **2.2** Implement `document_processor.py` using `libs.ocr.extract_text()`  
- [x] **2.3** Table extraction via Camelot / regex fallback  
- [x] **2.4** Support PDF & JPEG/PNG ≤15 MB  
- [x] **2.5** Multi-lang OCR: English, Hindi (add Tamil/Bengali stubs)  
- [x] **2.6** Document-type detection (hospital_bill vs pharmacy_invoice)  
- [x] **2.7** Robust error handling (corrupt files, OCR fail)  
- [x] **2.8** Tests with golden fixtures (coverage ≥ 90 %)

### 3. Rate Validation & Comparison Engine
- [x] **3.1** `reference_data_loader.py` – parse CGHS PDF → JSON, cache 24 h  
- [x] **3.2** Load ESI + state tariffs
- [x] **3.3** Integrate NPPA MRP CSV  
- [x] **3.4** Duplicate-test detection  
- [x] **3.5** Prohibited item detection via `data/prohibited.json`  
- [x] **3.6** `rate_validator.py` – compare & over-charge %, return deltas  
- [x] **3.7** `confidence_scorer.py` – combine rule weights + LLM fallback  
- [x] **3.8** State-specific validation using `state_code`
- [x] **3.9** Cache look-ups (`aiocache`, TTL 24 h)  
- [x] **3.10** Unit tests for all scenarios

### 4. API Service & Response Generation
- [x] **4.1** Build FastAPI app (`medbillguardagent.py`)  
- [x] **4.2** `POST /analyze` async endpoint (schema FR-017)  
- [x] **4.3** `/healthz` endpoint → `{ "status": "ok" }`  
- [x] **4.4** `/debug/example` processes fixture <10 s  
- [x] **4.5** Verdict logic: ok / warning / critical  
- [x] **4.6** Total over-charge (₹) calc  
- [x] **4.7** Markdown + SSML via `explanation_builder.py`  
- [x] **4.8** Next-steps suggestions (refund, ombudsman)  
- [x] **4.9** Plain-language counselling output  
- [x] **4.10** Latency tracking (`latency_ms`) + OTEL span  
- [x] **4.11** React-friendly JSON envelope (FR-019)  
- [x] **4.12** Idempotent by `doc_id` (Redis SETNX)  
- [x] **4.13** Structured error handling (RFC 7807)  
- [x] **4.14** Rate limiting via FastAPI-Limiter

### 5. Testing, Monitoring & Deployment
- [x] **5.1** Pytest suite ≥90 % coverage  
- [x] **5.2** `@timed_test(max_ms=150)` decorator on happy-path  
- [ ] **5.3** Integration tests with mocked OCR & LLM  
- [ ] **5.4** Golden fixture snapshot (`fixtures/example.pdf`)  
- [ ] **5.5** Locust load test (100 RPS, P95 ≤ 15 s)  
- [ ] **5.6** Structured `structlog` JSON logging  
- [ ] **5.7** Prometheus metrics (`service_latency_ms`, `error_total`)  
- [ ] **5.8** `/gdpr/export` + S3 30-day lifecycle rule  
- [ ] **5.9** Security headers + CORS config  
- [ ] **5.10** `SECURITY.md` with Trivy & Bandit guidelines  
- [ ] **5.11** `README.md` / `API.md` with cURL & JSON examples  
- [ ] **5.12** Verify container start <150 ms (CPU 300 m / RAM 512 Mi)  
- [ ] **5.13** End-to-end replay harness & final perf bench (≤12 s, ≥92 % recall)

---

**Stack:** Python 3.11, FastAPI, PyMuPDF, Tesseract, LanceDB, OpenAI/Mistral  
**Perf Targets:** ≤150 ms API overhead • ≤12 s doc processing • ≥92 % recall • <5 % FP