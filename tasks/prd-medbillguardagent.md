# Product Requirements Document: MedBillGuardAgent

## Introduction/Overview
The **MedBillGuardAgent** is a domain-specific AI micro-service designed to detect over-charges and unfair line-items in Indian hospital and diagnostic bills. Indian patients routinely pay **20-60 %** more than regulated CGHS / state-health rates without any verification mechanism.

The service will **auto-extract** every charge, **benchmark** against CGHS / ESI / state tariffs, **flag** discrepancies, and **generate refund letters + reminders**. This empowers citizens to defend against risky paperwork and claim legitimate refunds.

---

## Goals
1. **Primary Goal**: ≥92 % red-flag recall with <5 % false-positives  
2. **Performance**: ≤12 s median latency per 5-page bill  
3. **User Action**: ≥60 % click “Generate Refund Letter”  
4. **Compliance**: 100 % adherence to CGHS 2023 / ESI / state-scheme validation  
5. **Scalability**: Handle 15 MB uploads; English + Hindi (Bengali/Tamil roadmap)

---

## User Stories
### Primary Users
1. **Patient** – Upload bill → instant flags above CGHS rates.  
2. **Caregiver** – Snap pharmacy invoice → detect MRP over-billing.  
3. **TPA Officer** – Bulk-check 100 PDFs → auto-reject over-limit lines.  
4. **NGO Volunteer** – Get plain-language illegality explanation for counselling.

### Secondary User
5. **Healthcare Admin** – Audit bills vs. regulation to maintain compliance.

---

## Functional Requirements
### Core Processing
* **FR-001** Extract name, qty, unit, total from PDFs / JPEGs ≤15 MB.  
* **FR-002** English & Hindi OCR (Bengali/Tamil ready).  

### Validation
* **FR-004** Compare each line with CGHS 2023.  
* **FR-005** Validate against ESI + state tariffs.  
* **FR-006** Check pharmacy items vs. NPPA MRP CSV.  
* **FR-007** Detect duplicate tests.  
* **FR-008** Flag prohibited fees (e.g., “COVID kit”).

### Analysis & Reporting
* **FR-009** Calculate over-charge %.  
* **FR-010** Confidence per violation.  
* **FR-011** Markdown + SSML explanations.  
* **FR-012** Verdict: ok / warning / critical.  
* **FR-013** Total over-charge (₹).  

### User Guidance
* **FR-014** Suggest next actions (refund, ombudsman).  
* **FR-015** Generate PDF & DOCX refund letters (ActionAgent).  
* **FR-016** Plain-language counselling output.

### API & Integration
* **FR-017** REST endpoints with Pydantic schema.  
* **FR-018** `/healthz`, `/debug/example`.  
* **FR-019** Structured JSON for React dashboard.  
* **FR-020** WhatsApp image-share path.

### Performance & Reliability
* **FR-021** API compute ≤150 ms (excluding OCR).  
* **FR-022** OCR + LLM ≤12 s median.  
* **FR-023** Idempotent by `doc_id`.  
* **FR-024** Include `latency_ms` in response.

---

## Non-Goals (Out of Scope)
1. Document authenticity / watermarks (v2).  
2. Legal contract or loan analysis.  
3. Payment processing.  
4. Direct hospital-system integration.  
5. End-to-end insurance claim workflow.  
6. Multi-doc bundling in one request.  
7. Real-time tariff push (batch nightly OK).

---

## Technical Schema
### Request
```python
class MedBillGuardRequest(BaseModel):
    doc_id: str
    user_id: str
    s3_url: str
    document_type: str = "hospital_bill"
    state_code: str = "DL"

Response

class RedFlag(BaseModel):
    item: str
    billed: int
    max_allowed: int
    violation_type: str
    regulation_reference: str

class MedBillGuardResponse(BaseModel):
    doc_id: str
    verdict: str                # ok | warning | critical
    overcharge_total: int
    red_flags: list[RedFlag]
    summary: str
    next_steps: str
    confidence_score: float
    cites: list[str]
    latency_ms: int


⸻

Design Considerations

User Interface
	•	React dashboard tables + “Download Letter” / “Set Reminder”.
	•	WhatsApp flow returns concise JSON card.
	•	SSML for voice mode.

Data Sources
	•	Nightly-ingested CGHS PDF, ESI CSV, state tariffs.
	•	NPPA drug MRP CSV (weekly).

Architecture
	•	FastAPI async, NATS events.
	•	OCR: libs.ocr.extract_text() (PyMuPDF → PNG → Tesseract).
	•	LLM Provider interface with OpenAI GPT-4o fallback to local Mistral-7B.
	•	Vector RAG: LanceDB; k=4 tariff chunks.
	•	Caching: aiocache 24 h TTL on rate lookups.

Engineering Rules (extract)
	•	Contract-First schemas.
	•	MyPy strict, Ruff clean, Trivy no critical CVEs.
	•	@timed_test(max_ms=150) latency guard.
	•	Structured JSON logs (ts, level, svc, doc_id).
	•	Feature flags via Unleash.
	•	Idempotent: same doc_id → identical output.
	•	Event replay harness for nightly regression.

Security & Compliance
	•	AES-256 S3, 30-day auto-delete.
	•	Salted embeddings; no raw PII.
	•	/gdpr/export endpoint.
	•	Signed audit hashes (SHA-256) in ObjectLock bucket.

Performance
	•	Async HTTPX pool (50).
	•	Container limits: CPU 300 m, RAM 512 Mi.
	•	Locust load-test target: 100 RPS, P95 ≤ 15 s.

⸻

Success Metrics

Technical

Metric	Target
Recall	≥92 %
False-Positive	<5 %
Median Latency	≤8 s
Availability	99.5 %
Error Rate	<0.5 %

Business

Metric	Target
Action Conversion	≥60 %
Docs Processed	95 % success
User Confidence Avg	>0.8
Compliance Coverage	100 %

Operational

Metric	Target
Container CPU	<300 m
Memory	<512 Mi
Test Coverage	≥90 %
Trivy Critical CVEs	0
Startup Time	<150 ms


⸻

Open Questions
	1.	Rate-update frequency (daily vs weekly)?
	2.	Batch mode for TPAs in v1?
	3.	Next regional language priority?
	4.	Confidence threshold mapping warning vs critical?
	5.	Keep historical overcharge trends?
	6.	WhatsApp vs web launch sequence?
	7.	Handling conflicting rate sources?
	8.	User feedback loop for false-positives?

⸻

Implementation Priority

Phase	Duration	Deliverables
MVP (4 wks)	Core OCR, CGHS validation, REST API, PDF support	
v1.1 (2 wks)	Pharmacy invoices, NPPA MRPs, richer explanations, WA prep	
v1.2 (2 wks)	Multi-lang, bulk processing, advanced confidence, perf tuning	


⸻

Document Version: 1.0
Last Updated: June 2025
Target Delivery: Q1 2025

