# Clinical RAG API Server

Flask-based API server that connects the React frontend to the Clinical RAG system.

## Overview

This API server acts as a bridge between the React frontend and the Python-based RAG_chat_pipeline system. It provides endpoints for:

1. Sending chat messages to the RAG system
2. Retrieving available models
3. Serving the React frontend in production

## Setup and Installation

### Prerequisites

- Python 3.10+
- Existing RAG_chat_pipeline installation

### Installation

1. The required dependencies should already be installed if you've followed the main project setup. If not, install:

```bash
pip install flask flask-cors
```

2. Start the API server:

```bash
python api/app.py
```

## API Endpoints

### `POST /api/chat`

Process a chat message through the RAG system.

**Request Body:**

```json
{
  "message": "What medications were prescribed for patient 12345?",
  "chat_history": [
    {
      "role": "user",
      "content": "Previous message"
    },
    {
      "role": "assistant",
      "content": "Previous response"
    }
  ]
}
```

**Response:**

```json
{
  "response": "The patient was prescribed...",
  "sources": [
    {
      "content": "...",
      "metadata": {
        "hadm_id": "12345",
        "section": "prescriptions"
      }
    }
  ]
}
```

### `GET /api/models`

Get available embedding models and vector stores.

**Response:**

```json
{
  "embedding_models": ["ms-marco", "multi-qa", "mini-lm"],
  "vector_stores": ["faiss_mimic_sample1000_ms-marco", "faiss_mimic_sample1000_multi-qa"]
}
```

## Integration

This API server integrates with:

1. **RAG_chat_pipeline**: Uses the core functionality to process queries
2. **Frontend**: Serves API endpoints for the React application
3. **Models**: Accesses embedding models and vector stores defined in config
