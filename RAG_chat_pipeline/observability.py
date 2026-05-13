"""Application Insights wiring via the Azure Monitor OpenTelemetry distro.

Idempotent + opt-in: the function is safe to call multiple times and
silently no-ops when APPLICATIONINSIGHTS_CONNECTION_STRING is unset.
That keeps local development free of any cloud dependency while the
Azure Container Apps deployment (which sets the env var via Bicep)
gets traces, request metrics, and exception tracking automatically.

What you get when the env var IS set:
  - Auto-instrumentation of:
      Flask request/response (status, duration, route)
      requests, urllib3, urllib (covers the openai SDK's HTTP calls)
      Python `logging` module (logs flow into App Insights as traces)
  - Resource attributes: service.name = clinical-rag-api, version
  - The module exposes `tracer` and `meter` for callers that want
    explicit custom spans / counters (audit + claim-check are good
    candidates for follow-up instrumentation).

We deliberately don't instrument urllib http calls to localhost, the
Azure SDK telemetry endpoint, or the health probe path - those would
be self-noise.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import metrics, trace

logger = logging.getLogger(__name__)

_initialised = False
_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None

# Paths to keep out of the telemetry stream (would dwarf everything else)
_EXCLUDED_HTTP_TARGETS = ",".join([
    "/health",
])


def setup_observability(service_name: str = "clinical-rag-api") -> bool:
    """Configure Azure Monitor + auto-instrumentation.

    Returns True if telemetry was wired up, False if skipped (no
    connection string or already initialised). Caller decides whether
    to surface the result.
    """
    global _initialised, _tracer, _meter

    if _initialised:
        return False

    conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn_str:
        logger.info("Observability skipped: APPLICATIONINSIGHTS_CONNECTION_STRING not set.")
        return False

    # Configure the OTel auto-instrumentation. Setting the URL exclusion
    # before configure_azure_monitor so the env var is read by the
    # urllib/requests instrumentations at import time.
    os.environ.setdefault("OTEL_PYTHON_URLLIB_EXCLUDED_URLS", _EXCLUDED_HTTP_TARGETS)
    os.environ.setdefault("OTEL_PYTHON_REQUESTS_EXCLUDED_URLS", _EXCLUDED_HTTP_TARGETS)
    os.environ.setdefault("OTEL_PYTHON_FLASK_EXCLUDED_URLS", _EXCLUDED_HTTP_TARGETS)

    # Lazy import: only pay the import cost when telemetry is actually
    # being enabled. Keeps local-dev startup snappy.
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(
        connection_string=conn_str,
        resource_attributes={
            "service.name": service_name,
            "service.namespace": "msc-rag-v2",
        },
        enable_live_metrics=True,
    )

    _tracer = trace.get_tracer(service_name)
    _meter = metrics.get_meter(service_name)
    _initialised = True
    logger.info("Observability initialised for %s.", service_name)
    return True


def get_tracer() -> Optional[trace.Tracer]:
    """Tracer for callers that want to add explicit spans. None if disabled."""
    return _tracer


def get_meter() -> Optional[metrics.Meter]:
    """Meter for callers that want custom counters/histograms. None if disabled."""
    return _meter
