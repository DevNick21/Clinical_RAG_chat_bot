import hashlib
import logging
import os
import re
from typing import Any, Iterable, Optional


_ID_RE = re.compile(r"\b\d{6,10}\b")


def redact_text(text: str) -> str:
    return _ID_RE.sub("[REDACTED_ID]", text)


def summarize_text(text: Optional[str]) -> str:
    if not text:
        return "len=0"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"len={len(text)} sha={digest}"


def mask_ids(values: Optional[Iterable[Any]]) -> list[str]:
    if not values:
        return []
    masked = []
    for value in values:
        if value is None:
            continue
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8]
        masked.append(f"id:{digest}")
    return masked


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, int):
        value_str = str(value)
        return "[REDACTED_ID]" if _ID_RE.search(value_str) else value
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_value(v) for v in value]
    return value


class ClinicalLogger:
    LEVELS = {"quiet": 0, "error": 1, "warning": 2, "info": 3, "debug": 4}
    _configured = False
    _logger = logging.getLogger("clinical_rag")

    @classmethod
    def _configure(cls, level: Optional[str] = None) -> None:
        if cls._configured:
            return
        env_level = (level or os.getenv("CRAG_LOG_LEVEL", "info")).lower()
        logging_level = {
            "quiet": logging.ERROR,
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
        }.get(env_level, logging.INFO)

        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging_level,
                format="%(asctime)s %(levelname)s %(message)s",
            )
        cls._logger.setLevel(logging_level)
        cls._configured = True

    @classmethod
    def set_level(cls, level: str) -> None:
        cls._configure(level)
        logging_level = {
            "quiet": logging.ERROR,
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
        }.get(level, logging.INFO)
        cls._logger.setLevel(logging_level)

    @classmethod
    def _log(cls, level: int, msg: str, **meta: Any) -> None:
        cls._configure()
        if not cls._logger.isEnabledFor(level):
            return
        msg = redact_text(msg)
        if meta:
            sanitized = {k: _sanitize_value(v) for k, v in meta.items()}
            meta_str = " ".join(f"{k}={v}" for k, v in sanitized.items())
            cls._logger.log(level, f"{msg} | {meta_str}")
        else:
            cls._logger.log(level, msg)

    @classmethod
    def info(cls, msg: str, **meta: Any) -> None:
        cls._log(logging.INFO, msg, **meta)

    @classmethod
    def warning(cls, msg: str, **meta: Any) -> None:
        cls._log(logging.WARNING, msg, **meta)

    @classmethod
    def error(cls, msg: str, **meta: Any) -> None:
        cls._log(logging.ERROR, msg, **meta)

    @classmethod
    def success(cls, msg: str, **meta: Any) -> None:
        cls._log(logging.INFO, msg, **meta)

    @classmethod
    def debug(cls, msg: str, **meta: Any) -> None:
        cls._log(logging.DEBUG, msg, **meta)
