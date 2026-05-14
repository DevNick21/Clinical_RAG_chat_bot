"""Azure AI Foundry inference client factories.

Three model roles, each on its own Foundry deployment:

  * Answer model  (`get_llm`)        — reasoning model (gpt-5.4-nano @ low+).
                                       Captures reasoning summaries for
                                       clinical audit. Used for the
                                       streamed clinical QA answer.
  * Fast model    (`get_fast_llm`)   — small non-reasoning model
                                       (gpt-5.4-mini). Used for
                                       entity extraction + short
                                       follow-up rephrasing — both sit
                                       on the TTFP critical path so
                                       latency matters more than depth.
  * Audit model   (`get_audit_llm`)  — capable non-reasoning model
                                       (gpt-5-nano). Used for
                                       LLM-as-judge faithfulness checks
                                       on the final answer. Runs AFTER
                                       the stream, so it doesn't touch
                                       TTFP — quality is the priority.

All three target the same Foundry resource by default, so they share
`TARGET_URL` and `MODEL_API_KEY` unless explicitly overridden. Deployment
names are the per-role variable.
"""
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

from RAG_chat_pipeline.inference.responses_client import ResponsesAPIChatModel

load_dotenv()


def _required(name: str, value: Optional[str]) -> str:
    if not value:
        raise RuntimeError(
            f"Missing required env var: {name}. "
            "Set it in .env at the repo root."
        )
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_llm(
    reasoning_effort: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> ResponsesAPIChatModel:
    """Build the clinical-answer model (reasoning + audit-grade tracing).

    Required env vars:
      - TARGET_URL: Foundry OpenAI-compatible base URL (must end in /openai/v1)
      - MODEL_API_KEY: deployment key
      - MODEL_DEPLOYMENT_NAME: exact deployment name (e.g. gpt-5.4-nano)
    Optional:
      - REASONING_EFFORT: minimal|low|medium|high (default: low)
      - MAX_OUTPUT_TOKENS: int budget for reasoning + visible output
        combined. Default 16384, enough for medium effort. Bump for
        high effort or complex multi-step queries.
    """
    endpoint = _required("TARGET_URL", os.getenv("TARGET_URL"))
    api_key = _required("MODEL_API_KEY", os.getenv("MODEL_API_KEY"))
    deployment = _required("MODEL_DEPLOYMENT_NAME",
                           os.getenv("MODEL_DEPLOYMENT_NAME"))
    effort = reasoning_effort or os.getenv("REASONING_EFFORT", "low")

    if max_tokens is None:
        max_tokens = _int_env("MAX_OUTPUT_TOKENS", 16384)

    client = OpenAI(base_url=endpoint, api_key=api_key)
    return ResponsesAPIChatModel(
        client=client,
        model=deployment,
        reasoning_effort=effort,
        max_output_tokens=max_tokens,
    )


def get_fast_llm() -> ChatOpenAI:
    """Build the small non-reasoning model used on the TTFP critical path.

    Used by entity extraction (`extract_entities` LLM fallback) and short
    follow-up rephrasing (`ConversationManager._rephrase_safely`). Both
    are NER/condensation-shaped tasks where a small model performs
    indistinguishably from a reasoning model — at a fraction of the
    latency and token cost.

    Defaults to inheriting the answer model's Foundry resource (same
    endpoint + key), only overriding the deployment name. Provide
    `FAST_MODEL_BASE_URL` / `FAST_MODEL_API_KEY` to point at a different
    Foundry resource.

    Required env vars:
      - FAST_MODEL_DEPLOYMENT_NAME (e.g. gpt-5.4-mini, gpt-4o-mini)
    Optional:
      - FAST_MODEL_BASE_URL          (default: TARGET_URL)
      - FAST_MODEL_API_KEY           (default: MODEL_API_KEY)
      - FAST_MODEL_MAX_TOKENS        (default: 1024)
      - FAST_MODEL_TEMPERATURE       (default: 0.0 — these tasks want
                                     deterministic structured output)
    """
    endpoint = _required(
        "FAST_MODEL_BASE_URL or TARGET_URL",
        os.getenv("FAST_MODEL_BASE_URL") or os.getenv("TARGET_URL"),
    )
    api_key = _required(
        "FAST_MODEL_API_KEY or MODEL_API_KEY",
        os.getenv("FAST_MODEL_API_KEY") or os.getenv("MODEL_API_KEY"),
    )
    deployment = _required(
        "FAST_MODEL_DEPLOYMENT_NAME", os.getenv("FAST_MODEL_DEPLOYMENT_NAME")
    )
    max_tokens = _int_env("FAST_MODEL_MAX_TOKENS", 1024)
    temperature = float(os.getenv("FAST_MODEL_TEMPERATURE", "0.0"))

    return ChatOpenAI(
        base_url=endpoint,
        api_key=api_key,
        model=deployment,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=15,
    )


def get_audit_llm() -> ChatOpenAI:
    """Build the capable non-reasoning model used for LLM-as-judge audit.

    Used by `audit.claim_checker.check_claims_with_llm` (post-stream) to
    confirm or refute the regex-flagged sentences against the retrieved
    source documents. Doesn't sit on the TTFP path — quality matters
    more than latency here.

    Required env vars:
      - AUDIT_MODEL_DEPLOYMENT_NAME (e.g. gpt-5-nano, gpt-4.1, gpt-4o)
    Optional:
      - AUDIT_MODEL_BASE_URL          (default: TARGET_URL)
      - AUDIT_MODEL_API_KEY           (default: MODEL_API_KEY)
      - AUDIT_MODEL_MAX_TOKENS        (default: 2048)
      - AUDIT_MODEL_TEMPERATURE       (default: 0.0 — judge calls are
                                      classification-shaped)
    """
    endpoint = _required(
        "AUDIT_MODEL_BASE_URL or TARGET_URL",
        os.getenv("AUDIT_MODEL_BASE_URL") or os.getenv("TARGET_URL"),
    )
    api_key = _required(
        "AUDIT_MODEL_API_KEY or MODEL_API_KEY",
        os.getenv("AUDIT_MODEL_API_KEY") or os.getenv("MODEL_API_KEY"),
    )
    deployment = _required(
        "AUDIT_MODEL_DEPLOYMENT_NAME", os.getenv("AUDIT_MODEL_DEPLOYMENT_NAME")
    )
    max_tokens = _int_env("AUDIT_MODEL_MAX_TOKENS", 2048)
    temperature = float(os.getenv("AUDIT_MODEL_TEMPERATURE", "0.0"))

    return ChatOpenAI(
        base_url=endpoint,
        api_key=api_key,
        model=deployment,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
    )
