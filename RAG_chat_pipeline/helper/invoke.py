import re
from RAG_chat_pipeline.utils.logger import ClinicalLogger


def clean_llm_response(response):
    """
    Clean LLM responses by removing <think> tags and content
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
    Centralized LLM invocation with error handling and response cleaning
    """
    try:
        if hasattr(chain_or_llm, 'invoke'):
            response = chain_or_llm.invoke(input_data)
        else:
            # Direct LLM call
            response = chain_or_llm(input_data)

        # Auto-clean all model responses (remove think tags)
        response = clean_llm_response(response)

        return response

    except Exception as e:
        ClinicalLogger.error("LLM invoke error", context=context, error=str(e))
        return fallback_message
