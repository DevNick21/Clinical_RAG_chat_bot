"""TTFP / latency benchmark for the streaming /api/chat endpoint.

Measures three distinct moments per query, all wall-clock from the client side:

  * TTFB           — first SSE byte received from the server. With the
                     warmup event enabled this is dominated by retrieval +
                     pre-LLM work (NOT model reasoning).
  * TTFP           — Time To First (visible) Prediction = first SSE event
                     of `type: content` with non-empty payload. This is
                     the metric end-users feel.
  * total_ms       — full stream duration.

Also collects the server-side `timings` dict from the final metadata
event so we can see where the wall-clock went.

Run locally against `python -m platform.api.app` on :5000:

    python benchmarks/ttfp_bench.py \
        --url http://localhost:5000/api/chat \
        --api-key $env:API_KEY \
        --runs 10

Or against ACA:

    python benchmarks/ttfp_bench.py \
        --url https://<aca-fqdn>/api/chat \
        --api-key <bearer> \
        --runs 10 \
        --out benchmarks/results/ttfp_after.json

Compare before/after by running with the optimisations off then on:

    setx CRAG_ENABLE_LLM_ENTITY_EXTRACTION true
    setx CRAG_ENABLE_REPHRASING true
    setx CRAG_ENABLE_STREAMING_WARMUP_EVENT false
    (restart the API, run, save as ttfp_before.json)

    setx CRAG_ENABLE_LLM_ENTITY_EXTRACTION false
    setx CRAG_ENABLE_REPHRASING false
    setx CRAG_ENABLE_STREAMING_WARMUP_EVENT true
    (restart the API, run, save as ttfp_after.json)

The harness intentionally uses urllib (stdlib only) so it has no
project-dependency overhead.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


# Mix of patterns we want to cover:
#  - explicit hadm_id  → regex extraction wins, no LLM pre-call
#  - explicit section  → regex catches section keyword
#  - free-form         → no IDs / sections, exercises the LLM-fallback path
#  - follow-up         → exercises the chat-context rephrasing path
DEFAULT_PROMPTS: List[Dict[str, Any]] = [
    {
        "label": "filtered_hadm_id",
        "message": "What diagnoses are recorded for admission 25282710?",
        "chat_history": [],
    },
    {
        "label": "filtered_meds",
        "message": "What medications were prescribed for admission 25282710?",
        "chat_history": [],
    },
    {
        "label": "global_freeform",
        "message": "Summarise common patterns in critical care admissions.",
        "chat_history": [],
    },
    {
        "label": "short_followup",
        "message": "What about the labs?",
        "chat_history": [
            {"role": "user", "content": "Show me diagnoses for admission 25282710"},
            {"role": "assistant", "content": "Diagnoses include hypertension, sepsis, ..."},
        ],
    },
]


def _run_one(url: str, api_key: str, prompt: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """Send one streaming request, return per-stage timings."""
    body = json.dumps({
        "message": prompt["message"],
        "chat_history": prompt.get("chat_history", []),
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
        },
    )

    t0 = time.perf_counter()
    ttfb: Optional[float] = None
    ttfp: Optional[float] = None
    documents_found: Optional[int] = None
    pre_llm_ms_server: Optional[float] = None
    server_timings: Dict[str, Any] = {}
    error: Optional[str] = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # First byte = first chunk of the response body. urllib's
            # read(1) blocks until the server flushes something, which
            # under SSE is the moment any data is yielded.
            buffer = b""
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
                buffer += chunk
                # Process complete SSE events
                while b"\n\n" in buffer:
                    event, buffer = buffer.split(b"\n\n", 1)
                    for line in event.split(b"\n"):
                        if not line.startswith(b"data: "):
                            continue
                        try:
                            data = json.loads(line[len(b"data: "):].decode("utf-8"))
                        except json.JSONDecodeError:
                            continue
                        kind = data.get("type")
                        if kind == "status" and data.get("stage") == "retrieval_done":
                            documents_found = data.get("documents_found")
                            pre_llm_ms_server = data.get("pre_llm_ms")
                        elif kind == "content" and data.get("content") and ttfp is None:
                            ttfp = time.perf_counter() - t0
                        elif kind == "metadata":
                            server_timings = (data.get("metadata") or {}).get("timings", {})
                        elif kind == "error":
                            error = data.get("content")
    except urllib.error.URLError as e:
        error = f"transport: {e}"
    except Exception as e:  # noqa: BLE001 — bench tool, want everything
        error = f"unexpected: {type(e).__name__}: {e}"

    total = time.perf_counter() - t0
    return {
        "label": prompt["label"],
        "ttfb_ms": round(ttfb * 1000, 1) if ttfb is not None else None,
        "ttfp_ms": round(ttfp * 1000, 1) if ttfp is not None else None,
        "total_ms": round(total * 1000, 1),
        "documents_found": documents_found,
        "pre_llm_ms_server": pre_llm_ms_server,
        "server_timings": server_timings,
        "error": error,
    }


def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-label p50/p95 over successful runs."""
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    out: Dict[str, Any] = {}
    for label, runs in by_label.items():
        ok = [r for r in runs if r["error"] is None]
        def _pct(field: str, p: float) -> Optional[float]:
            xs = [r[field] for r in ok if r.get(field) is not None]
            if not xs:
                return None
            xs.sort()
            idx = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
            return xs[idx]
        out[label] = {
            "runs": len(runs),
            "errors": sum(1 for r in runs if r["error"] is not None),
            "ttfb_p50_ms": _pct("ttfb_ms", 0.50),
            "ttfb_p95_ms": _pct("ttfb_ms", 0.95),
            "ttfp_p50_ms": _pct("ttfp_ms", 0.50),
            "ttfp_p95_ms": _pct("ttfp_ms", 0.95),
            "total_p50_ms": _pct("total_ms", 0.50),
            "total_p95_ms": _pct("total_ms", 0.95),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True, help="Full URL of the streaming /api/chat endpoint")
    ap.add_argument("--api-key", required=True, help="Bearer API key (API_KEY env on server)")
    ap.add_argument("--runs", type=int, default=5, help="Iterations per prompt (default 5)")
    ap.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout seconds")
    ap.add_argument("--prompts", type=str, default=None,
                    help="Optional JSON file with a list of {label, message, chat_history}")
    ap.add_argument("--out", type=str, default=None, help="Write the full result JSON here")
    args = ap.parse_args()

    if args.prompts:
        with open(args.prompts, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    else:
        prompts = DEFAULT_PROMPTS

    rows: List[Dict[str, Any]] = []
    print(f"Running {len(prompts)} prompts x {args.runs} runs against {args.url}", file=sys.stderr)
    for run_idx in range(args.runs):
        for prompt in prompts:
            row = _run_one(args.url, args.api_key, prompt, timeout=args.timeout)
            row["run_index"] = run_idx
            rows.append(row)
            print(
                f"  [{prompt['label']}#{run_idx}] "
                f"ttfb={row['ttfb_ms']}ms ttfp={row['ttfp_ms']}ms "
                f"total={row['total_ms']}ms"
                + (f" ERROR={row['error']}" if row["error"] else ""),
                file=sys.stderr,
            )

    summary = _summarise(rows)
    out_json = {
        "url": args.url,
        "runs_per_prompt": args.runs,
        "rows": rows,
        "summary": summary,
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2)
        print(f"\nWrote {args.out}", file=sys.stderr)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
