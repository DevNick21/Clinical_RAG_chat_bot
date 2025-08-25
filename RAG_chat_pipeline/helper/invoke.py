from RAG_chat_pipeline.config.config import LLM_MODEL
import re
from RAG_chat_pipeline.config.config import LLM_MODEL

import sys
from pathlib import Path

# Add the project root to the path to import from RAG_chat_pipeline
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def clean_deepseek_response(response):
    """
    Clean DeepSeek R1 responses by removing <think> tags and content
    """
    if not isinstance(response, str):
        return response

    # Remove <think>...</think> blocks (including multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Remove standalone <think> or </think> tags
    cleaned = re.sub(r'</?think>', '', cleaned)

    # Clean up extra whitespace and newlines
    # Multiple newlines to double
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    cleaned = cleaned.strip()

    return cleaned


def safe_llm_invoke(chain_or_llm, input_data, fallback_message="Error generating response", context="LLM operation"):
    """
    Centralized LLM invocation with error handling and DeepSeek response cleaning
    """
    try:
        if hasattr(chain_or_llm, 'invoke'):
            response = chain_or_llm.invoke(input_data)
        else:
            # Direct LLM call
            response = chain_or_llm(input_data)

        # Auto-clean DeepSeek responses
        if "deepseek" in LLM_MODEL.lower():
            response = clean_deepseek_response(response)

        return response

    except Exception as e:
        print(f" {context} Error: {e}")
        return fallback_message
