from setuptools import setup, find_packages

setup(
    name="RAG_chat_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core LangChain Framework
        "langchain==0.3.25",
        "langchain-community==0.3.24",
        "langchain-core==0.3.63",
        "langchain-huggingface==0.2.0",
        "langchain-ollama==0.3.3",
        "langchain-text-splitters==0.3.8",

        # Vector Storage & Embeddings
        "faiss-cpu==1.11.0",
        "sentence-transformers==4.1.0",

        # Machine Learning & Deep Learning
        "torch==2.7.1",
        "transformers==4.52.4",
        "safetensors==0.5.3",
        "tokenizers==0.21.1",
        "tiktoken==0.9.0",
        "huggingface-hub==0.32.4",
        "scikit-learn==1.6.1",

        # Data Processing & Analysis
        "pandas==2.3.0",
        "numpy==2.2.6",
        "matplotlib==3.10.1",
        "seaborn==0.13.2",
        "plotly",
        "tqdm==4.67.1",

        # Database & SQL
        "duckdb==1.3.1",
        "sqlalchemy==2.0.41",

        # LLM Integration
        "ollama==0.5.1",

        # API Server
        "flask==3.0.2",
        "flask-cors==4.0.1",
    ],
    author="DevNick21",
    description="Clinical RAG system for MIMIC-IV data analysis",
    python_requires=">=3.10",
)
