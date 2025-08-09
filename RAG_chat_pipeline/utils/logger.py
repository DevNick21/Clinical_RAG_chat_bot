"""Lightweight centralized logger with simple verbosity levels.
Use ClinicalLogger.set_level(...) at app startup to control verbosity.
"""


class ClinicalLogger:
    """Centralized logging utility with verbosity control"""
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
            print(f"ℹ️ {msg}")

    @staticmethod
    def warning(msg):
        if ClinicalLogger._enabled("warning"):
            print(f"⚠️ {msg}")

    @staticmethod
    def error(msg):
        if ClinicalLogger._enabled("error"):
            print(f"❌ {msg}")

    @staticmethod
    def success(msg):
        if ClinicalLogger._enabled("info"):
            print(f"✅ {msg}")

    @staticmethod
    def debug(msg):
        if ClinicalLogger._enabled("debug"):
            print(f"🔍 {msg}")
