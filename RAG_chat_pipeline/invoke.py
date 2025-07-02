def safe_llm_invoke(chain_or_llm, input_data, fallback_message="Error generating response", context="LLM operation"):
    """
    Centralized LLM invocation with error handling
    """
    try:
        if hasattr(chain_or_llm, 'invoke'):
            return chain_or_llm.invoke(input_data)
        else:
            # Direct LLM call
            return chain_or_llm(input_data)
    except Exception as e:
        print(f"⚠️ {context} Error: {e}")
        return fallback_message
