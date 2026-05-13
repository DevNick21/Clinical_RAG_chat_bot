"""
API Server for Clinical RAG Chat
Connects React frontend to the RAG_chat_pipeline backend.

Hardening (Phase E):
  - Bearer-token auth on /api/* via @require_api_key (key in API_KEY env).
    /health stays open so orchestrators can probe without credentials.
  - CORS restricted to ALLOWED_ORIGINS (comma-separated). Default empty
    -> no cross-origin browser access; same-origin and server-to-server
    are unaffected.
  - flask-limiter per-IP rate limits, in-memory backend (sufficient with
    gunicorn --workers 1; swap to redis when scaling out).
  - debug mode env-gated. FLASK_DEBUG=true to enable, off by default.
"""

import os
import json
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from RAG_chat_pipeline.observability import setup_observability

# Wire telemetry BEFORE any other RAG_chat_pipeline import or Flask
# instantiation so the OpenTelemetry instrumentation can patch
# Flask + requests + urllib3 cleanly. No-ops locally without
# APPLICATIONINSIGHTS_CONNECTION_STRING; live in ACA via the Bicep
# env wiring.
setup_observability(service_name="clinical-rag-api")

from RAG_chat_pipeline.core.main import main as initialize_clinical_rag
from RAG_chat_pipeline.utils.logger import ClinicalLogger, summarize_text, mask_ids
from RAG_chat_pipeline.config.config import model_names, vector_stores

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/build')

# CORS - env-configured. Empty list = no cross-origin requests allowed.
# Set ALLOWED_ORIGINS=http://localhost:3000,https://my-frontend.example.com
_allowed_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()
]
CORS(app, origins=_allowed_origins or [], supports_credentials=False)

# Rate limiter. memory:// is fine while we run gunicorn with --workers 1;
# multi-worker / multi-replica deploys need a shared backend (redis).
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"],
    storage_uri="memory://",
)


def require_api_key(fn):
    """Bearer-token auth. Reads expected key from API_KEY env at request time.

    Fail-closed: if API_KEY isn't configured, every protected request
    returns 503 with a clear reason. Better than silently allowing
    unauthenticated traffic in a misconfigured deploy.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        expected = os.getenv("API_KEY")
        if not expected:
            return jsonify({"error": "API_KEY not configured on server"}), 503
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[len("Bearer "):].strip() != expected:
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


# Initialize RAG system
ClinicalLogger.info("Initializing Clinical RAG System")
try:
    chatbot = initialize_clinical_rag()
    ClinicalLogger.info("Clinical RAG System initialized successfully")
except Exception as e:
    ClinicalLogger.error("Error initializing Clinical RAG System", error=str(e))
    chatbot = None


@app.route('/health', methods=['GET'])
@limiter.exempt
def health():
    """Liveness + readiness probe for ACA / load balancers.

    Returns 200 only when the RAG bot finished initialising (embedding
    model + FAISS + chunked docs all loaded). 503 otherwise so the
    orchestrator holds traffic until we're warm. Exempt from auth and
    rate limits so probes can always run.
    """
    if chatbot is None:
        return jsonify({"status": "unready", "reason": "chatbot init failed or in progress"}), 503
    return jsonify({"status": "ok"}), 200


@app.route('/api/chat', methods=['POST'])
@require_api_key
@limiter.limit("10 per minute")
def chat():
    """Handle streaming chat requests (default)"""
    if not chatbot:
        return jsonify({
            'error': 'RAG system not initialized'
        }), 500

    data = request.json
    if not data or 'message' not in data:
        return jsonify({
            'error': 'Missing message in request'
        }), 400

    user_message = data['message']
    chat_history = data.get('chat_history', [])

    ClinicalLogger.info(
        "API streaming chat request",
        message=summarize_text(user_message),
        chat_history_len=len(chat_history) if chat_history else 0,
        chat_history_ids=mask_ids([item.get("id") for item in chat_history if isinstance(item, dict)]),
    )

    def generate():
        """Generator function for streaming responses"""
        try:
            chunk_count = 0
            for chunk in chatbot.chat_stream(user_message, chat_history):
                chunk_count += 1
                
                # Format as Server-Sent Events
                if 'error' in chunk:
                    yield f"data: {json.dumps({'type': 'error', 'content': chunk['error'], 'done': True})}\n\n"
                    break
                elif 'content' in chunk:
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk['content'], 'done': chunk.get('done', False)})}\n\n"
                elif 'done' in chunk and chunk['done']:
                    # Send final metadata
                    metadata = chunk.get('metadata', {})
                    yield f"data: {json.dumps({'type': 'metadata', 'metadata': metadata, 'done': True})}\n\n"
                    break
                
                # Periodic flush for better streaming experience
                if chunk_count % 5 == 0:
                    yield ""  # Empty line to ensure flush
            
            # Ensure stream ends properly
            yield f"data: {json.dumps({'type': 'end', 'done': True})}\n\n"
            
        except Exception as e:
            ClinicalLogger.error("Error in streaming chat", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'content': 'Streaming error', 'done': True})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/api/chat/non-streaming', methods=['POST'])
@require_api_key
@limiter.limit("10 per minute")
def chat_non_streaming():
    """Handle non-streaming chat requests (fallback)"""
    if not chatbot:
        return jsonify({
            'error': 'RAG system not initialized'
        }), 500

    data = request.json
    if not data or 'message' not in data:
        return jsonify({
            'error': 'Missing message in request'
        }), 400

    user_message = data['message']
    chat_history = data.get('chat_history', [])

    ClinicalLogger.info(
        "API non-streaming chat request",
        message=summarize_text(user_message),
        chat_history_len=len(chat_history) if chat_history else 0,
        chat_history_ids=mask_ids([item.get("id") for item in chat_history if isinstance(item, dict)]),
    )

    # Process with RAG system
    try:
        response = chatbot.chat(user_message, chat_history)

        ClinicalLogger.info(
            "API non-streaming chat response",
            response_len=len(str(response)),
        )

        return jsonify({
            'response': response,
            'sources': chatbot.sources if hasattr(chatbot, 'sources') else []
        })
    except Exception as e:
        ClinicalLogger.error("Error processing message", error=str(e))
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/models', methods=['GET'])
@require_api_key
def get_models():
    """Return available models"""
    return jsonify({
        'embedding_models': list(model_names.keys()),
        'vector_stores': list(vector_stores.keys())
    })


@app.route('/api/sample-suggestions', methods=['GET'])
@require_api_key
def get_sample_suggestions():
    """Return sample query suggestions with real data"""
    try:
        from RAG_chat_pipeline.utils.data_provider import get_sample_data
        import random

        sample_data = get_sample_data()
        if not sample_data:
            # Fallback suggestions if data loading fails
            return jsonify({
                'suggestions': [
                    "What diagnoses does patient 10000032 have?",
                    "Show me lab results for admission 25282710",
                    "What medications were prescribed for patient 10006508?"
                ]
            })

        # Get random sample of HADM IDs
        hadm_ids = sample_data['hadm_ids']
        random_hadm_ids = random.sample(hadm_ids, min(10, len(hadm_ids)))

        # Create varied suggestions using real data
        suggestion_templates = [
            "What diagnoses are recorded for admission {}?",
            "Show me lab results for admission {}",
            "What medications were prescribed for admission {}?",
            "What procedures were performed during admission {}?",
            "Show me microbiology results for admission {}",
            "What transfers occurred during admission {}?",
            "Tell me about the patient demographics for admission {}",
            "What are the vital signs recorded for admission {}?",
            "Show me pharmacy records for admission {}",
            "What services were involved in admission {}?"
        ]

        # Generate suggestions with random HADM IDs
        suggestions = []
        for i, template in enumerate(suggestion_templates):
            hadm_id = random_hadm_ids[i % len(random_hadm_ids)]
            suggestions.append(template.format(hadm_id))

        return jsonify({
            'suggestions': suggestions
        })

    except Exception as e:
        ClinicalLogger.error("Error getting sample suggestions", error=str(e))
        # Fallback suggestions
        return jsonify({
            'suggestions': [
                "What diagnoses does patient 10000032 have?",
                "Show me lab results for admission 25282710",
                "What medications were prescribed for patient 10006508?"
            ]
        })

# Serve React static files in production


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    # Direct invocation is dev-only; production runs under gunicorn (see
    # Dockerfile CMD). debug=True is off unless explicitly enabled via
    # FLASK_DEBUG to avoid the auto-reloader / interactive debugger in
    # any non-dev context.
    port = int(os.environ.get('PORT', 5000))
    debug = os.getenv("FLASK_DEBUG", "").lower() in ("true", "1", "yes")
    app.run(host='0.0.0.0', port=port, debug=debug)
