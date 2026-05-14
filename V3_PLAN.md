# V3 Plan — Tests + CI/CD

Self-contained handoff for a fresh session. Read this top-to-bottom; do not assume any prior conversation context.

## Where v2 landed (last session, pushed to origin/v2 as of f99f00d)

**Working production deploy on Azure ACA**, image `clinical-rag:06a802c`, talking to Azure AI Foundry (gpt-5-nano answer + gpt-5-mini fast + gpt-5-nano audit). Recent commits:

| SHA | What |
|---|---|
| `f99f00d` | **Injectable-LLM refactor** — `ClinicalRAGBot(*, llm=..., fast_llm=..., audit_llm=...)`. The only precondition for V3 tests; already done. |
| `06a802c` | Pinned `cachetools==5.3.2` + `openai==1.109.1` (caught at deploy as a crash-on-import bug — they were transitive deps that the slim container didn't install). |
| `fc60d3e` | TTFP three-model roles + LLM-as-judge audit + early SSE warmup. |
| `d52d2b6` `d599a48` `13f7450` | Bicep → Terraform conversion (state imported live, zero-drift apply). |
| (earlier) | God-class decomposition: [retriever.py](RAG_chat_pipeline/core/retriever.py), [content_processor.py](RAG_chat_pipeline/core/content_processor.py), [conversation_manager.py](RAG_chat_pipeline/core/conversation_manager.py). Orchestrator down to 641 lines (was 1422). |

**Test state today: zero tests, no `pyproject.toml`, no `tests/` directory.** This is the V3 starting point.

## V3 anchor

**Tests + CI/CD** — full plan, ~9-12 days. Decided in the previous session because:

- Zero tests is the biggest production gap (just hit a real crash-on-import bug at deploy that tests would have caught at PR-time).
- The v2 god-class decomposition was the precondition. The four core modules are now small + injectable enough to test cleanly.
- Strongest interview narrative: "I refactored to enable TDD, then added the safety net."

User explicitly chose **full plan over MVP cut** and **tiny real FAISS fixture over pure mocks**.

## Phase breakdown

| Phase | Scope | Days | Status |
|---|---|---|---|
| **0. Injectable-LLM refactor** | `ClinicalRAGBot.__init__` accepts `llm` / `fast_llm` / `audit_llm` kwargs via `_UNSET` sentinel | 0.5 | ✅ done (`f99f00d`) |
| **1. Test infra** | `pyproject.toml` (ruff+mypy+pytest), `tests/{unit,integration,fixtures}/`, `conftest.py`, `FakeLLM`, **tiny real FAISS fixture vectorstore** (10-20 docs, committed to git) | 1-2 | ⏳ next |
| **2. Unit tests** | ≥ 80% coverage per module. Order: `ContentProcessor` (easy, pure-stateless) → `Retriever` → `ConversationManager` → `entity_extraction` → `claim_checker` | 2-3 | pending |
| **3. Integration tests** | `ClinicalRAGBot.clinical_search` + `chat_stream` end-to-end with stubbed LLM. All 4 paths: filtered / global / preflight / fallback | 2 | pending |
| **4. Lint + types** | ruff clean repo-wide; mypy strict on `core/` + `inference/` initially | 1 | pending |
| **5. GitHub Actions CI** | `ci.yml`: lint → test → coverage gate. SP creds via GH Secrets | 1 | pending |
| **6. CD** | `deploy.yml`: on push to v2 → docker build → ACR push → ACA revision update | 1-2 | pending |
| **7. Refactor surfaced by testing** | Whatever else needs injection (`DataProvider`?), surface during test-writing | 1 | pending |

## Phase 1 detailed task list (start here)

### Step 1.1 — `pyproject.toml` (~30 min)

Single source of truth for ruff + mypy + pytest config. Don't add a `setup.py` or `setup.cfg` — keep one file. Suggested skeleton:

```toml
[project]
name = "clinical-rag"
version = "0.0.0"
requires-python = ">=3.11"

[tool.ruff]
line-length = 100
target-version = "py311"
extend-exclude = ["venv", "platform/frontend/node_modules", "models", "vector_stores"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "B", "UP"]  # pyflakes, pycodestyle, isort, pep8-naming, bugbear, pyupgrade
ignore = ["E501"]  # line-length is enforced by formatter

[tool.mypy]
python_version = "3.11"
strict = false  # phase 4 flips this true on core/ + inference/ only

[[tool.mypy.overrides]]
module = ["RAG_chat_pipeline.core.*", "RAG_chat_pipeline.inference.*"]
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --cov=RAG_chat_pipeline --cov-report=term-missing"
markers = [
    "slow: tests that load real FAISS / sentence-transformers (~3s+)",
    "integration: tests that wire the full ClinicalRAGBot together",
    "live: tests that hit real Azure Foundry (skip in CI by default)",
]

[tool.coverage.run]
omit = ["*/tests/*", "venv/*", "report/*", "benchmarks/*", "data_engineering/*"]
```

### Step 1.2 — directory layout (~10 min)

```
tests/
  __init__.py
  conftest.py                 # shared fixtures (see 1.4)
  fixtures/
    build_fixture_vectorstore.py    # one-shot script (1.5)
    mini_faiss/                     # generated artifact (committed)
      index.faiss
      index.pkl
    mini_chunked_docs.pkl
  unit/
    __init__.py
    test_content_processor.py
    test_retriever.py
    test_conversation_manager.py
    test_entity_extraction.py
    test_claim_checker.py
  integration/
    __init__.py
    test_clinical_rag_bot.py
    test_chat_stream.py
```

### Step 1.3 — `FakeLLM` stub (~30 min)

Lives in `tests/conftest.py`. Mirrors the public surface `ClinicalRAGBot` actually touches:

```python
class FakeLLM:
    """Drop-in for the Foundry Responses-API LLM in tests.

    Records call args, returns canned responses. No network, no Azure.
    """
    def __init__(self, response="canned answer (Admission 12345678, Diagnoses)"):
        self.response = response
        self.calls = []
        self.last_reasoning_summaries = []
        self.last_response_id = None

    def invoke(self, prompt):
        self.calls.append(("invoke", prompt))
        return type("Resp", (), {"content": self.response})()

    def stream(self, prompt):
        self.calls.append(("stream", prompt))
        yield self.response
```

`ConversationManager` and `safe_llm_invoke` both go through `.invoke(...)`. Streaming chains call `.stream(...)`. Confirm the surface by grepping `self.llm.` and `self.fast_llm.` across `core/`.

### Step 1.4 — `conftest.py` shared fixtures (~1 hr)

Required fixtures:

```python
@pytest.fixture(scope="session")
def fixture_chunked_docs():
    """Load tests/fixtures/mini_chunked_docs.pkl (10-20 Document objects covering all sections)."""

@pytest.fixture(scope="session")
def fixture_clinical_emb():
    """Real PubMedBERT-style embedder. Module-scoped — load once per test session (~5s)."""

@pytest.fixture(scope="session")
def fixture_vectorstore(fixture_clinical_emb):
    """Load tests/fixtures/mini_faiss/ via FAISS.load_local. ~200ms."""

@pytest.fixture
def fake_llm():
    return FakeLLM()

@pytest.fixture
def bot(fixture_vectorstore, fixture_clinical_emb, fixture_chunked_docs, fake_llm):
    """Fully-stubbed ClinicalRAGBot. No Azure calls."""
    return ClinicalRAGBot(
        fixture_vectorstore, fixture_clinical_emb, fixture_chunked_docs,
        llm=fake_llm, fast_llm=fake_llm, audit_llm=None,
    )
```

### Step 1.5 — fixture vectorstore (~2 hr)

Critical decision: **build a real tiny FAISS index from a 10-20 doc subset of the actual MIMIC sample**, commit it to git. Not pure mocks. User explicitly chose this in the previous session because:
- Real cosine sim catches embedder-boundary bugs
- Module-scoped load is ~200ms — negligible
- One-time setup, then static

Implementation: `tests/fixtures/build_fixture_vectorstore.py` reads the existing `mimic_sample_1000/chunked_docs.pkl`, picks a deterministic subset (e.g., 5 admissions × ~3 chunks each covering diagnoses + labs + prescriptions), embeds them with the same PubMedBERT used in prod, saves the FAISS index + a `mini_chunked_docs.pkl` to `tests/fixtures/`. The script is committed; the artifacts it produces are also committed (so CI doesn't have to re-embed).

Pick admissions that exercise:
- Single-admission queries (one ID present)
- Multi-admission comparison (two IDs both present)
- ID-mismatch preflight (an ID NOT in the fixture, like `99999999`)
- Section filtering (at least one admission with diagnoses + labs)

## Files to read first when restarting

In this order — they refresh context fastest:

1. **[CLAUDE.md](CLAUDE.md)** — developer profile, communication preferences, what NOT to do
2. **[V3_PLAN.md](V3_PLAN.md)** (this file)
3. **[RAG_chat_pipeline/core/clinical_rag.py](RAG_chat_pipeline/core/clinical_rag.py)** — orchestrator, see how the new `_UNSET` injection works (lines ~36-95)
4. **[RAG_chat_pipeline/core/retriever.py](RAG_chat_pipeline/core/retriever.py)** — easiest module to test second (after content_processor)
5. **[RAG_chat_pipeline/core/content_processor.py](RAG_chat_pipeline/core/content_processor.py)** — pure stateless, **start tests here for the easy win**
6. **[RAG_chat_pipeline/core/conversation_manager.py](RAG_chat_pipeline/core/conversation_manager.py)** — note `fast_llm` is the LLM consumer, not `llm`
7. **[RAG_chat_pipeline/inference/azure_client.py](RAG_chat_pipeline/inference/azure_client.py)** — `get_llm` / `get_fast_llm` / `get_audit_llm` signatures (you may need to mock these via `monkeypatch.setattr` in addition to the kwarg injection)
8. **[requirements.txt](requirements.txt)** — add `pytest`, `pytest-cov`, `pytest-asyncio`, `ruff`, `mypy` (probably as `[project.optional-dependencies] dev = [...]` in `pyproject.toml` rather than polluting prod requirements)

## Constraints + gotchas

- **PHI redaction.** [RAG_chat_pipeline/utils/logger.py](RAG_chat_pipeline/utils/logger.py) masks 6-10 digit numeric IDs in log output. Tests using IDs like `12345678` will see `[REDACTED_ID]` in captured logs. Don't assert on exact log contents that contain IDs.
- **Cost.** All testing should mock the LLM. Never write a test that hits real Foundry without a `@pytest.mark.live` marker that's skipped by default.
- **Azure auth in CI.** Phase 5/6 will need a service principal stored as a GitHub Secret with `Contributor` on `rg=msc_project` and `AcrPush` on `mscragv2acr`. Use OIDC federation (no client secret) — see [GitHub docs](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-azure).
- **Don't break the prod deploy.** `main.py` calls `ClinicalRAGBot(vs, emb, docs)` positionally — already-verified back-compat after the `_UNSET` refactor. Don't change the positional signature.
- **Streaming generator tests.** `chat_stream()` is a generator. Drain with `list(bot.chat_stream(...))` and assert on chunk shapes (`{"content": ...}`, `{"done": True, "metadata": ...}`).

## Open questions to resolve early in V3

1. Where does test data live? Options:
   - Subset of `mimic_sample_1000/chunked_docs.pkl` (committed) — recommended
   - Synthetic docs generated by the fixture script — less realistic but smaller diff
2. Coverage gate threshold — start at 60% per module and ratchet up, or 80% from day one?
3. Mypy strict on which packages? Plan says `core/` + `inference/`. `audit/` and `helper/` arguably belong too.
4. Pre-commit hook? Run ruff on commit. Cheap to add, catches formatting before push.

## Out of scope for V3

These belong elsewhere:
- Benchmark expansion (separate project — see memory note `project_benchmark_expansion_direction.md`)
- TTFP measurement harness (`benchmarks/ttfp_bench.py` already exists from parallel work)
- Frontend test coverage (Jest/RTL on the React app — could be a V3.1)
- Migration to FastAPI (was in original v2 plan, then dropped; revisit only if needed)
