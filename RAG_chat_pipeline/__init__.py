"""
RAG Chat Pipeline Package
Clinical RAG system for MIMIC-IV data analysis
"""

# Prefer absolute import to avoid analyzer issues
try:
    from RAG_chat_pipeline.utils.logger import ClinicalLogger  # type: ignore
except Exception:  # pragma: no cover
    # Minimal fallback to prevent hard crashes during analysis/runtime
    class ClinicalLogger:
        LEVELS = {"quiet": 0, "error": 1, "warning": 2, "info": 3, "debug": 4}
        level = "info"

        @classmethod
        def set_level(cls, level: str):
            if level in cls.LEVELS:
                cls.level = level

        @classmethod
        def _enabled(cls, lvl: str) -> bool:
            return cls.LEVELS.get(cls.level, 3) >= cls.LEVELS.get(lvl, 3)

        @staticmethod
        def info(msg):
            if ClinicalLogger._enabled("info"):
                print(f" {msg}")

        @staticmethod
        def warning(msg):
            if ClinicalLogger._enabled("warning"):
                print(f" {msg}")

        @staticmethod
        def error(msg):
            if ClinicalLogger._enabled("error"):
                print(f" {msg}")

        @staticmethod
        def success(msg):
            if ClinicalLogger._enabled("info"):
                print(f" {msg}")

        @staticmethod
        def debug(msg):
            if ClinicalLogger._enabled("debug"):
                print(f" {msg}")

__version__ = "1.0.0"
__author__ = "Ekene Iheanacho"
__description__ = "Clinical RAG system with evaluation framework"

__all__ = ["ClinicalLogger"]
