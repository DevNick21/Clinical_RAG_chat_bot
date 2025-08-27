"""
API Server for Clinical RAG Chat
Connects React frontend to the RAG_chat_pipeline backend
"""

from RAG_chat_pipeline.core.main import main as initialize_clinical_rag
from RAG_chat_pipeline.config.config import model_names, vector_stores
import sys
import os
from pathlib import Path
import json
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

# Add project root to path BEFORE importing any project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# NOW import RAG system components (after path is set)

# Import RAG system components

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/build')
CORS(app)  # Enable CORS for all routes

# Initialize RAG system
print(" Initializing Clinical RAG System...")
try:
    chatbot = initialize_clinical_rag()
    print(" Clinical RAG System initialized successfully")
except Exception as e:
    print(f" Error initializing Clinical RAG System: {e}")
    chatbot = None


@app.route('/api/chat', methods=['POST'])
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

    # Debug: Log incoming request details
    print(f" API Streaming Chat Request:")
    print(f"  - Message: '{user_message}'")
    print(f"  - Chat history length: {len(chat_history) if chat_history else 0}")
    if chat_history:
        print(f"  - Last 2 history items: {chat_history[-2:] if len(chat_history) >= 2 else chat_history}")

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
            print(f"Error in streaming chat: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'Streaming error: {str(e)}', 'done': True})}\n\n"

    return Response(
        generate(),
        mimetype='text/plain',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'POST'
        }
    )


@app.route('/api/chat/non-streaming', methods=['POST'])
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

    # Debug: Log incoming request details
    print(f" API Non-Streaming Chat Request:")
    print(f"  - Message: '{user_message}'")
    print(f"  - Chat history length: {len(chat_history) if chat_history else 0}")

    # Process with RAG system
    try:
        response = chatbot.chat(user_message, chat_history)

        # Debug: Log response details
        print(f" API Non-Streaming Chat Response:")
        print(f"  - Response length: {len(str(response))}")
        print(f"  - Response preview: {str(response)[:200]}...")

        return jsonify({
            'response': response,
            'sources': chatbot.sources if hasattr(chatbot, 'sources') else []
        })
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Return available models"""
    return jsonify({
        'embedding_models': list(model_names.keys()),
        'vector_stores': list(vector_stores.keys())
    })


@app.route('/api/sample-suggestions', methods=['GET'])
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
        print(f"Error getting sample suggestions: {e}")
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
