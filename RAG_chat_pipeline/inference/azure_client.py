"""Azure AI Foundry inference client factory.

Returns a Responses-API-backed chat model that captures reasoning summaries
for clinical audit. Reads endpoint, key, deployment name, and reasoning
effort from environment (.env at repo root).
"""
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from RAG_chat_pipeline.inference.responses_client import ResponsesAPIChatModel

load_dotenv()


def get_llm(
    reasoning_effort: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> ResponsesAPIChatModel:
    """Build a ResponsesAPIChatModel pointed at Azure AI Foundry.

    Required env vars:
      - TARGET_URL: Foundry OpenAI-compatible base URL (must end in /openai/v1)
      - MODEL_API_KEY: deployment key
      - MODEL_DEPLOYMENT_NAME: exact deployment name (e.g. gpt-5-nano)
    Optional:
      - REASONING_EFFORT: minimal|low|medium|high (default: low)
      - MAX_OUTPUT_TOKENS: int budget for reasoning + visible output
        combined. Default 16384, enough for medium effort. Bump for
        high effort or complex multi-step queries.
    """
    endpoint = os.getenv("TARGET_URL")
    api_key = os.getenv("MODEL_API_KEY")
    deployment = os.getenv("MODEL_DEPLOYMENT_NAME")
    effort = reasoning_effort or os.getenv("REASONING_EFFORT", "low")

    # Resolve max_tokens: explicit arg > env var > default
    if max_tokens is None:
        env_val = os.getenv("MAX_OUTPUT_TOKENS")
        if env_val:
            try:
                max_tokens = int(env_val)
            except ValueError:
                max_tokens = 16384
        else:
            max_tokens = 16384

    missing = [
        name
        for name, val in [
            ("TARGET_URL", endpoint),
            ("MODEL_API_KEY", api_key),
            ("MODEL_DEPLOYMENT_NAME", deployment),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"Missing required env vars: {', '.join(missing)}. "
            "Set them in .env at the repo root."
        )

    client = OpenAI(base_url=endpoint, api_key=api_key)
    return ResponsesAPIChatModel(
        client=client,
        model=deployment,
        reasoning_effort=effort,
        max_output_tokens=max_tokens,
    )
