"""
Benchmarking and evaluation components
"""

# Define available modules for lazy importing to avoid circular import issues
__all__ = [
    "ClinicalRAGEvaluator",
    "ModelEvaluationRunner",
    "EvaluationResultsManager",
    "generate_gold_questions_from_data"
]


def __getattr__(name):
    """Lazy import to avoid circular import issues when modules are run with -m"""
    if name == "ClinicalRAGEvaluator":
        from .rag_evaluator import ClinicalRAGEvaluator
        return ClinicalRAGEvaluator
    elif name == "ModelEvaluationRunner":
        from .model_evaluation_runner import ModelEvaluationRunner
        return ModelEvaluationRunner
    elif name == "EvaluationResultsManager":
        from .evaluation_results_manager import EvaluationResultsManager
        return EvaluationResultsManager
    elif name == "generate_gold_questions_from_data":
        from .gold_questions import generate_gold_questions_from_data
        return generate_gold_questions_from_data
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
