# AGENTS.md — Clinical RAG Production Refactor

## ⚠️ IMPORTANT: Context Limitation

The architecture plans in the root (plan_01 through plan_05) were produced in a Codex.ai planning session that only had access to 4 source files:

- `main.py`
- `invoke.py`
- `clinical_rag.py` (partial — reviewed via GitHub, not full paste)
- `config.py`

**Before implementing any plan, read every file in the repo first.** The plans contain assumptions about code structure, method names, and behaviour that may be wrong. Validate each assumption against the actual code before acting on it. Flag any conflicts between the plans and reality.

Files the planning session did NOT see (non-exhaustive):
- `app.py` (Flask API)
- `entity_extraction.py`
- `embeddings_manager.py` / `load_or_create_vectorstore`
- Prompt templates
- Any utility/helper files
- `requirements.txt` / `pyproject.toml`
- Vectorstore creation and storage logic
- Evaluation scripts

---

## Developer Context — Who You're Working With

### Background & Experience Level
- **Started coding:** 2022. Self-taught JavaScript (frontend), then self-taught Python (primary language now).
- **Formal education:** Electronics & Computer Engineering (ECE) undergrad (Nigeria), MSc in AI with Distinction (Manchester Metropolitan University).
- **Current role:** GenAI Engineer at Betfred (production systems: behavioural anomaly detection, OCR compliance monitoring).
- **AI tool usage:** Uses AI to code heavily. This is normal — calibrate accordingly. Don't assume hand-written code; do assume the developer understands what the code does at the level indicated below.

### This Codebase Specifically
- **Clinical RAG chatbot** — MSc dissertation project, MIMIC-IV dataset.
- **AI usage:** ~95% AI-assisted code generation.
- **Understanding level:** ~95%. This is the highest-understanding project. The developer spent significant time on it, tested 54 model combinations (9 embedding × 6 LLM), built BioBERT evaluation, understands the RAG pipeline deeply.
- **What this means:** You can discuss architectural decisions, trade-offs, and implementation details without over-explaining. The developer will follow. But always verify shared assumptions — high understanding ≠ memorised every line.

### Other Projects (for calibrating analogies and references)
Use these when explaining new concepts — relate to what's already known:

| Project | AI Use | Understanding | Useful For Analogies |
|---------|--------|---------------|---------------------|
| Clinical RAG (this repo) | 95% | 95% | RAG pipelines, embeddings, LLM integration, evaluation |
| Agentic Product Crawler | 90% | 90% | Web scraping, agent loops, self-healing schemas |
| BinaryTreeTextAnalyzer (C#) | 90% | 90% | Data structures, algorithms, typed languages |
| WhackAMole QLearning | Partial | 100% | RL concepts, reward shaping, training loops |
| React Blog Project | No AI | 100% | React components, state, routing, frontend patterns |
| Mini Till System (CV) | 65% | 65% | Computer vision basics — but don't assume deep CV knowledge |
| FileScanner | 30% | 30% | Don't reference this — understanding is low |

### Technical Strengths (lean on these)
- **Python:** Strong. LangChain, PyTorch, FastAPI, pandas, async patterns.
- **RAG/LLM systems:** Deep. Embeddings, vector stores, retrieval strategies, prompt engineering, evaluation metrics.
- **Production systems:** Real experience at Betfred — compliance monitoring, anomaly detection in regulated industry.
- **Physics/Electronics/CS fundamentals:** Strong high-school level. Good for hardware-software analogies (Arduino fire detection systems, IR sensors, circuit thinking).
- **Interdisciplinary thinking:** High instinct for cross-domain analogies. Architecture (building) → architecture (software), circuits → data pipelines, etc. Use these when explaining new concepts.
- **Engineering mindset:** ECE background means systems thinking comes naturally — inputs, outputs, feedback loops, failure modes.

### Technical Gaps (calibrate around these)
- **Pure mathematics:** Weaker. Above average in further maths but don't assume comfort with heavy mathematical notation or proofs. Explain math concepts intuitively first, formally second.
- **Pure electrical engineering:** Weak despite ECE degree. Don't use deep EE analogies.
- **C#/.NET ecosystem:** Used it once (BinaryTreeTextAnalyzer) with AI. Don't assume fluency.
- **DevOps/Infrastructure:** Learning. Docker basics understood, AWS being learned through this project. Explain infrastructure concepts clearly — don't assume prior Terraform/ECS/ALB experience.
- **Low-level systems programming:** Not a strength. Avoid C/Rust/memory-management analogies.

### Communication Preferences (follow these strictly)

1. **Brief by default.** Don't over-explain. If more detail is needed, the developer will ask.
2. **Structured output.** Use clear headers, tables, numbered steps when presenting information. No wall-of-text dumps.
3. **Relate to known things.** When introducing a new concept, connect it to a project or library the developer already knows (see table above). Example: "tenacity retry works like the retry logic you'd build into your Agentic Crawler's request loop, but declarative."
4. **No information dumps.** Break complex topics into digestible pieces. Present the overview first, then offer to go deeper on specific parts.
5. **Question assumptions.** The developer has a tendency to think they're right. Push back when something seems off. Double-check your own responses too — don't just agree.
6. **Ask clarifying questions first.** Before answering anything non-trivial, ask for full context and edge cases that could affect the answer. This is a hard rule.
7. **Code standards:**
   - Prioritise performance AND readability (comments, clear naming).
   - Use library algorithms where they exist — mention which library and why.
   - Minimise raw code count with libraries — mention where they reduce complexity.
   - Code should be optimal at scale and fail-safe.
   - Always consider: what happens when this fails? What's the fallback?
8. **Confidence scoring.** On non-trivial claims, indicate confidence level. Break problems into smaller pieces, check from multiple angles, fix weak reasoning before committing.
9. **Search before answering.** If web search or documentation lookup would ground the answer with more accuracy, do it before responding.

### Thinking Style
- **Interdisciplinary pattern matcher.** Learns best through analogies to known domains. Building architecture → software architecture. Circuit design → data flow. Assembly line → pipeline processing.
- **Practical over theoretical.** Show working code, then explain why. Not the other way around.
- **Bottom-up learner.** Understands by building, not by reading theory first. TDD approach in this project aligns with this naturally.

---

## Project Goal

Productionise the existing Clinical RAG system (MSc project, MIMIC-IV dataset) from research code into an actual production-grade system. Not "portfolio production" (looks nice on GitHub) — **actual production** (handles real users, doesn't fall over, recovers from failures, observable, secure).

### Dual Purpose
This project serves two goals simultaneously:
1. **Production system:** A genuinely deployable clinical RAG chatbot with proper infrastructure, monitoring, and fault tolerance.
2. **Interview talking points:** The developer is actively interviewing for AI/ML roles (Anaplan GenAI Engineer, Scale AI, LGC, HMLR, others). Every architectural decision, trade-off, and production concern addressed here becomes concrete interview material. When making decisions, note what makes them "interview-worthy" — the reasoning matters as much as the implementation.

**Key interview themes this project demonstrates:**
- God class → modular architecture (refactoring at scale)
- Research code → production code (what changes and why)
- Dual inference with failover (system design thinking)
- TTFT optimisation from 30s → <200ms (performance engineering)
- TDD in a real project (not just "I know what TDD is")
- Observability from day one (Prometheus, structured logging, alerting)
- Cost-conscious infrastructure decisions (spot instances, Bedrock fallback)

---

## Architecture Decisions (Confirmed)

### Web Framework
- **Flask → FastAPI** — async for non-blocking LLM calls, Pydantic validation, auto Swagger docs, proper SSE streaming

### Inference
- **Dual paths:** vLLM (self-hosted on g5.xlarge spot) + AWS Bedrock (Llama 3.1 8B managed)
- **InferenceRouter** with auto-failover: vLLM primary → Bedrock fallback
- **LLMClient Protocol** — any provider implements same interface, swappable
- Local dev uses Bedrock API (no GPU needed)

### TTFT Optimisation
- Core engineering differentiator: 30s baseline → target <200ms
- vLLM tuning levers: `--max-model-len`, prefix caching, AWQ quantization, `--gpu-memory-utilization`
- Benchmark suite comparing vLLM vs Bedrock TTFT

### Configuration
- **Global mutable `config.py` → Pydantic Settings** (`pydantic-settings`)
- Immutable after creation, env var loading with `CRAG_` prefix, thread-safe

### Logging
- **`print()` everywhere → structlog** — structured JSON logs in prod, coloured console in dev
- Request ID middleware for tracing

### Caching
- **Hand-rolled `_EmbCache` → `cachetools.LRUCache`** — O(1), thread-safe

### Retry Logic
- **Bare `try/except` → tenacity** — exponential backoff, configurable retries, transient vs permanent error differentiation

### Monitoring
- **Prometheus + Grafana** — TTFT histograms, error counters, health gauges, failover tracking
- Two dashboards: Operations Overview + TTFT Deep Dive
- Alert rules for TTFT spikes, error rate, inference availability

### Testing
- **Zero tests → pytest + pytest-asyncio + pytest-cov**
- TDD approach: write tests first, implement until green
- 80% overall coverage gate, 90%+ on core/

### API Hardening
- Pydantic request/response validation
- API key authentication
- slowapi rate limiting
- Restricted CORS origins
- No `debug=True`

### Infrastructure
- Multi-stage Dockerfile
- docker-compose: API + Prometheus + Grafana
- AWS ECS Fargate + ALB
- GitHub Actions CI/CD: lint (ruff + mypy) → test → build → deploy

### Code Quality
- **ruff** (replaces flake8+isort+black)
- **mypy** strict mode

## Known Issues in Current Code (From Audit)

These were identified from partial code review. **Verify each one against the actual code.**

1. `ClinicalRAGBot` — 700+ line god class, 25+ methods, does everything
2. `sys._getframe(1)` used to detect caller (API vs CLI) — replace with explicit parameters
3. `_has_hallucination()` returns False, disabled — delete dead code
4. `_EmbCache` has O(n) `.remove()` — replace with cachetools
5. `question_embedding` potentially undefined in >100 doc branch of `_semantic_search_on_docs` — verify and fix
6. `config.py` `set_models()` mutates module-level globals — not thread-safe
7. `try/except pass` in config.py on import — silent failure
8. `app.py` uses `print()`, `CORS(app)` with no restrictions, `debug=True`, global chatbot init
9. `entity_extraction.py` regex `\b(\d{8})\b` matches dates as hadm_ids
10. `invoke.py` has duplicate import, `sys.path` hack, no retries/timeouts
11. Streaming endpoint uses `mimetype='text/plain'` but sends SSE format

## Plan Documents

Located in `/docs/production_plans/`:

| Document | Contents |
|----------|----------|
| `plan_01_code_refactoring.md` | Class decomposition, migration map (which method goes where), dead code list, library additions, new project structure |
| `plan_02_tdd_test_structure.md` | Every test file, shared fixtures, unit/integration/benchmark test examples, pytest config, coverage targets |
| `plan_03_infrastructure.md` | Dockerfile, docker-compose, ECS Fargate, ALB config, GitHub Actions CI/CD, cost estimates |
| `plan_04_monitoring.md` | structlog setup, Prometheus metrics definitions, Grafana dashboard layouts, PromQL queries, alert rules, health endpoints |
| `plan_05_execution_timeline.md` | 8-phase dependency graph, task-level estimates, TDD build order, MVP cut lines, risk mitigations |

## Implementation Order

Follow the dependency graph in plan_05:

```
Phase 1: Foundation (settings, logging, schemas) — 2 days
Phase 2: Core Decomposition (retriever, extractor, processor) — 4 days
Phase 3: Inference Layer (protocol, clients, router, retry) — 3 days
Phase 4: Orchestrator (thin RAGEngine) — 1 day
Phase 5: API Layer (FastAPI, routes, middleware) — 3 days
Phase 6: Monitoring (Prometheus, Grafana) — 2 days
Phase 7: Infrastructure (Docker, AWS, CI/CD) — 3 days
Phase 8: TTFT Optimisation (vLLM tuning, benchmarks) — 2 days
```

**TDD throughout:** For each module, write the test file first → run (all fail) → implement until green → refactor.

## Key Libraries

```
fastapi, uvicorn, slowapi                    # Web framework
pydantic-settings                            # Config
structlog                                    # Logging
langchain-aws, langchain-openai              # LLM clients (Bedrock, vLLM)
httpx, tenacity                              # Async HTTP, retry
cachetools                                   # Embedding cache
prometheus-client, prometheus-fastapi-instrumentator  # Monitoring
pytest, pytest-asyncio, pytest-cov           # Testing
ruff, mypy                                   # Code quality
```

## What NOT to Do

- Don't bolt infrastructure onto the current code — refactor first
- Don't keep `sys._getframe()` — use dependency injection
- Don't keep `print()` — use structlog everywhere
- Don't keep global mutable config — use Pydantic Settings
- Don't keep the god class — decompose per plan_01
- Don't skip tests — TDD is the approach, not optional
- Fine-tuning is a **separate project** — not in scope here
- Evaluation framework branches to a **separate repo** — not in scope here
- Don't over-explain code to the developer — they'll ask if they need more
- Don't dump all information at once — break it down, offer to go deeper

---

## Progress & Memory (Updated 2026-03-01)

### ✅ Iteration 1: Comprehensive Codebase Audit

**Completed:**
1. Read full `clinical_rag.py` (1,169 lines) — identified all 24 methods & dependencies
2. Created `AUDIT_AND_PLAN.md` — validated all 11 known issues, updated timeline
3. Created `CLINICAL_RAG_DECOMPOSITION_MAP.md` — detailed method-to-module mapping for Phase 2-5

**Key Findings:**
- ✅ ClinicalRAGBot is decomposable without major blockers (no circular dependencies found)
- ✅ sys._getframe(1) used in 3 methods (ask_question, chat, chat_stream) — will remove in Phase 4
- ✅ _has_hallucination() disabled (returns False) — can safely delete
- ✅ _EmbCache with O(n) remove() — will replace with cachetools
- ✅ SECTION_RULES hardcoded — will move to config.settings
- ✅ 13 config imports — will consolidate to Pydantic Settings
- ✅ LLM hardcoded as Ollama — will become Bedrock/vLLM via InferenceRouter
- ⚠️ No Flask app.py exists (mentioned in AGENTS.md, but doesn't exist) — will create properly in Phase 5

**Timeline Updated:**
- Original: 20 days
- With buffer for 52KB code complexity: 21 days
- **MVP (Phases 1-5):** 13 days
- **Full production (Phases 1-8):** 21 days

**Next:** Begin Phase 1 with TDD task list

### ✅ Iteration 2: Phase 1 Planning & Task List

**Completed:**
1. Created `PHASE_1_FOUNDATION_TASKS.md` — detailed TDD task list for 5 tasks
2. Task 1.1: Project setup (pyproject.toml, ruff, mypy, pytest)
3. Task 1.2: Settings module (Pydantic Settings with validation)
4. Task 1.3: Logging setup (structlog for dev/prod)
5. Task 1.4: API schemas (request/response Pydantic models)
6. Task 1.5: Test fixtures (conftest.py)

**Phase 1 Duration:** 2 days (estimated)
- Task 1.1 (Setup): 2 hours
- Task 1.2 (Settings): 3 hours
- Task 1.3 (Logging): 2 hours
- Task 1.4 (Schemas): 2 hours
- Task 1.5 (Fixtures): 1 hour
- **Total: 10 hours**

**Phase 1 Gate (must all pass):**
```bash
pytest tests/unit/ -v --cov=RAG_chat_pipeline --cov-fail-under=85
mypy RAG_chat_pipeline/ --strict
ruff check RAG_chat_pipeline/
```

**Deliverables:** 600 lines of new code (config, logging, schemas, tests)

**Ready to Start:** Yes — Phase 1 is fully planned and ready for TDD implementation

### Documents Created

| Document | Purpose | Status |
|----------|---------|--------|
| AUDIT_AND_PLAN.md | Codebase audit + updated timeline | ✅ Complete |
| CLINICAL_RAG_DECOMPOSITION_MAP.md | Method-by-method decomposition mapping | ✅ Complete |
| PHASE_1_FOUNDATION_TASKS.md | TDD task list for Phase 1 | ✅ Complete |
| This section | Memory/progress tracker | 📝 In progress |