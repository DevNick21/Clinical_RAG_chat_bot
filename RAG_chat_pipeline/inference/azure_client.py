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
    max_tokens: int = 2048,
) -> ResponsesAPIChatModel:
    """Build a ResponsesAPIChatModel pointed at Azure AI Foundry.

    Required env vars:
      - TARGET_URL: Foundry OpenAI-compatible base URL (must end in /openai/v1)
      - MODEL_API_KEY: deployment key
      - MODEL_DEPLOYMENT_NAME: exact deployment name (e.g. gpt-5-nano)
    Optional:
      - REASONING_EFFORT: minimal|low|medium|high (default: low)
    """
    endpoint = os.getenv("TARGET_URL")
    api_key = os.getenv("MODEL_API_KEY")
    deployment = os.getenv("MODEL_DEPLOYMENT_NAME")
    effort = reasoning_effort or os.getenv("REASONING_EFFORT", "low")

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
