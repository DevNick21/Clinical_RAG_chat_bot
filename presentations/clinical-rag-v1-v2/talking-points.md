# RealityMine Clinical RAG V1/V2 Reveal Deck - Rehearsal Notes

Target time: 7 minutes. Safe range: 5-10 minutes.

Open speaker view with `S` while presenting. Export with `?print-pdf` if you need a PDF.

## 1. Clinical RAG

Core line:

"This was my MSc dissertation project: a Retrieval-Augmented Generation chatbot over MIMIC-IV clinical records. I am presenting it in two stages: V1, the research prototype, and V2, the production refactor."

Emphasise:

- It is not just an AI chatbot.
- It retrieves clinical evidence first, then generates an answer.
- It demonstrates technical understanding, data-quality thinking, and testing depth.

## 2. EHR friction and natural-language access

Core line:

"EHR systems are powerful, but they have a high barrier to entry. This project reduces that barrier by letting the user ask in natural language, retrieving the relevant clinical evidence first, then answering with citations."

Data-quality angle:

- Ease of use is a product requirement, not just a UI preference.
- Retrieval quality and citation quality matter as much as API correctness.
- The product should fail clearly when it cannot find the requested admission or section.

## 3. Product view

Core line:

"From a product perspective, this is not just an AI chatbot. It is a workflow tool: the user wants to move from a natural-language clinical question to a trustworthy, evidence-backed answer faster."

Mention:

- User: researcher or analyst working across long clinical records.
- Product value: lower EHR friction plus evidence, not fluent text alone.
- Requirements: natural-language entry, evidence-first retrieval, cited answer, safe failure, and traceable data-quality checks.

RealityMine bridge:

"This matches RealityMine's product world because the value is not the final screen alone. The value is the trustworthy data journey behind it."

## 4. V1 research prototype

Core line:

"V1 proved the clinical-domain retrieval design: MIMIC-IV data, metadata-aware document formation, HADM/section query parsing, custom retrieval, local Ollama generation, and a simple UI."

Mention:

- Python, LangChain-style RAG flow, FAISS, Ollama, Flask, React.
- Documents kept the metadata that made retrieval useful: `hadm_id` and `section`.
- The custom retriever parses query signals, filters candidate documents, then re-ranks semantically.
- The objective was research validation, not production readiness.

## 5. 54-combo model bake-off

Core line:

"The evaluation work was not just manual prompting. I tested 9 embedding models against 6 language models, giving 54 model combinations and 1,080 evaluations."

Interaction:

- Hover or click the heatmap cells while rehearsing.
- The selected panel should start on `multi-qa + phi3` with F1 `0.758`.

Strong interview phrase:

"An AI system can pass normal code tests but still fail from a user perspective if retrieval is weak or the answer is unsupported."

## 6. What worked and what cracked

Core line:

"V1 worked as a dissertation prototype, but it exposed production weaknesses."

Use 3 examples only:

- Large class with too many responsibilities.
- Slow responses, with median response time around 62 seconds in the evaluation stats.
- Missing production controls such as auth, rate limits, audit trail, and observability.

Do not over-apologise. Say it like an engineer:

"That is exactly what prototypes are useful for: they reveal the next set of engineering problems."

## 7. V2 production refactor

Core line:

"V2 kept the RAG idea but strengthened the system around it."

Mention:

- Hardened Flask + gunicorn.
- Bearer auth, restricted CORS, rate limiting.
- Blob-backed data and FAISS index.
- Azure Foundry Responses API.
- Audit logs, claim checker, telemetry, Terraform, Azure Container Apps.

Important correction:

Do not say V2 is FastAPI unless you actually migrate it before the interview. Current repo reality is hardened Flask.

## 8. V2 request journey

Core line:

"V2 turns the answer into a traceable request journey: client, API edge, query parsing, custom retrieval, inference, and audit."

RealityMine bridge:

"RealityMine thinks in journeys across digital touchpoints. This is a different domain, but the engineering habit is similar: follow the event through the system and prove each stage behaved correctly."

Mention:

- API edge: auth, CORS, rate limits.
- Query parser: regex first, LLM fallback only when needed.
- Retrieval: metadata filter by `hadm_id` / `section`, then semantic re-rank.
- Inference: Azure Foundry Responses API.
- Audit: citations, claim checker and JSON trace.

## 9. V1 vs V2

Core line:

"V1 answered 'can metadata-aware RAG work?' V2 answers 'can it be operated, traced, and tested?'"

Data-quality and testing angle:

- Each V2 improvement creates a testable contract.
- Auth should fail closed.
- Health should reflect readiness.
- Streaming should end cleanly.
- Audits should preserve citations as `hadm_id` and `section`.

## 10. Product quality gates

Core line:

"For a data engineer, I would defend product contracts rather than just click through the UI."

Good examples:

- Usability contract: natural-language questions resolve to the right intent.
- Access contract: MIMIC-IV-derived data stays controlled, authenticated, and auditable.
- Retrieval contract: known `hadm_id` and section queries retrieve expected evidence.
- Citation contract: claims map back to admission and section evidence.
- Ops contract: auth, health checks, audit IDs, and logs support diagnosis.

## 11. What the user sees

Core line:

"The UI looks simple, but V2 adds traceability behind every answer."

Mention:

- `/health` endpoint.
- True SSE streaming.
- Audit ID per request.
- Claim checker for unsupported specifics.

## 12. How I would support this as a data product

Core line:

"I would support and test the data journey: ease of use, query parsing, retriever accuracy, citation correctness, and access plus operations."

Good examples:

- Natural-language queries reduce the need to understand EHR internals.
- Query parser extracts admission IDs and section intent.
- Known admission ID plus section retrieves expected documents.
- Every answer claim has an admission/section citation.
- Auth fails closed and restricted data access is auditable.
- SSE stream produces content events and a final done event.
- Cold start downloads the FAISS index and health checks pass.

## 13. RealityMine fit

Core line:

"RealityMine's work is about privacy-first measurement of real digital behaviour. My project is clinical AI, but the transferable skill is evidence-led data quality and testing across a journey."

Mention:

- Journey thinking: request lifecycle mirrors behavioural journey thinking.
- Privacy-first data work: clinical records demand careful traceability.
- Testing beyond UI: visible output is only the last surface.

## 14. Closing

Core line:

"The biggest learning was moving from 'does it answer?' to 'can I trust the whole system journey?'"

Close with:

"For RealityMine, this project shows the same habit across data, AI, software, product quality, testing, and operations: look beyond the visible output, follow the journey, preserve evidence, observe failures, and catch issues early."

## Likely interview questions

### What was your own contribution?

"I did more than wire up a generic RAG pipeline. I programmatically shaped retrieval for the clinical domain: document formation preserved `hadm_id` and `section`, query/entity extraction pulled those signals from natural language, the custom retriever filtered by metadata before semantic re-ranking, and then I evaluated model combinations and handled V2 production hardening with auth, deployment, audit logging, and observability."

### How did you test answer quality?

"I compared model combinations using retrieval/generation evaluation metrics, then looked at whether answers were grounded in retrieved clinical context. In V2 I added audit and claim-checking so answer quality is more traceable."

### What was the hardest challenge?

"The hardest part was that AI failures are not always obvious exceptions. The code can run, but retrieval can be weak or the generated answer can include unsupported specifics. I handled that by adding evaluation, audit trails, and claim checking."

### What would you improve next?

"I would expand automated tests around retrieval regression, add stronger LLM-as-judge evaluation for semantic drift, and improve load testing around streaming and cold starts."

### Why does this fit RealityMine?

"RealityMine works with privacy-first behavioural data and user journeys. My project is clinical AI, but the transferable habit is the same: trace the journey, preserve the evidence, test the quality gates, and do not assume the final output is trustworthy unless the upstream data path is trustworthy."
