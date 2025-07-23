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
from flask import Flask, request, jsonify, send_from_directory
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
print("🚀 Initializing Clinical RAG System...")
try:
    chatbot = initialize_clinical_rag()
    print("✅ Clinical RAG System initialized successfully")
except Exception as e:
    print(f"❌ Error initializing Clinical RAG System: {e}")
    chatbot = None


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
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

    # Process with RAG system
    try:
        response = chatbot.chat(user_message, chat_history)
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
