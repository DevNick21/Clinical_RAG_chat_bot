#!/bin/bash
# ============================================================================
# Clinical RAG System - Environment Setup Script
# ============================================================================
# Automated setup script for the MIMIC-IV Clinical RAG system
# This script creates a Python virtual environment and installs dependencies
# Last updated: July 2025

set -e  # Exit on any error

echo "🏥 Clinical RAG System - Environment Setup"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    print_error "Please install Python 3.11 or later first"
    print_error "Visit: https://www.python.org/downloads/"
    exit 1
fi

# Determine Python command
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

print_success "Python found: $($PYTHON_CMD --version)"

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
print_status "Python version: $PYTHON_VERSION"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_error "Requirements file 'requirements.txt' not found"
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment already exists
if [ -d "venv" ]; then
    print_warning "Virtual environment 'venv' already exists"
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing virtual environment..."
        rm -rf venv
        print_status "Creating new virtual environment..."
        $PYTHON_CMD -m venv venv
    else
        print_status "Using existing virtual environment"
    fi
else
    print_status "Creating new virtual environment 'venv'..."
    $PYTHON_CMD -m venv venv
fi

print_success "Virtual environment ready!"

# Activate environment and install dependencies
print_status "Activating virtual environment and installing dependencies..."
source venv/bin/activate

# Upgrade pip to latest version
print_status "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
print_status "Installing project dependencies..."
pip install -r requirements.txt

print_success "Dependencies installed!"

# Verify key packages
print_status "Verifying package installation..."
python -c "
import sys
print(f'Python version: {sys.version}')

packages = [
    ('langchain', 'LangChain'),
    ('faiss', 'FAISS'),
    ('sentence_transformers', 'Sentence Transformers'),
    ('pandas', 'Pandas'),
    ('torch', 'PyTorch')
]

for module, name in packages:
    try:
        __import__(module)
        print(f'✓ {name} imported successfully')
    except ImportError as e:
        print(f'✗ {name} import failed: {e}')
"

print_success "Environment verification completed!"

echo
echo "🎉 Setup Complete!"
echo "=================="
echo "Your clinical RAG virtual environment is ready to use."
echo
echo "To activate the environment manually:"
echo "  source venv/bin/activate"
echo
echo "To deactivate the environment:"
echo "  deactivate"
echo
echo "To start Jupyter notebook:"
echo "  source venv/bin/activate"
echo "  jupyter notebook"
echo
echo "To run the main RAG pipeline:"
echo "  source venv/bin/activate"
echo "  python RAG_chat_pipeline/main.py"
echo

# Optional: Create activation script
read -p "Would you like to create an 'activate-rag.sh' script to activate this environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat > activate-rag.sh << 'EOF'
#!/bin/bash
# Activate Clinical RAG Environment
source venv/bin/activate
echo "🏥 Clinical RAG environment activated!"
echo "Python: $(python --version)"
echo "Virtual Environment: $(which python)"
EOF
    chmod +x activate-rag.sh
    print_success "Created 'activate-rag.sh' activation script"
    print_status "Run './activate-rag.sh' to activate the environment"
fi

# Optional: Add alias to bash profile
read -p "Would you like to add an alias 'rag-env' to activate this environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ALIAS_LINE="alias rag-env='cd \"$(pwd)\" && source venv/bin/activate'"
    
    # Check if alias already exists
    if grep -q "alias rag-env" ~/.bashrc 2>/dev/null; then
        print_warning "Alias 'rag-env' already exists in ~/.bashrc"
    else
        echo "" >> ~/.bashrc
        echo "# Clinical RAG Environment Alias" >> ~/.bashrc
        echo "$ALIAS_LINE" >> ~/.bashrc
        print_success "Added alias 'rag-env' to ~/.bashrc"
        print_status "Run 'source ~/.bashrc' or restart your terminal to use the alias"
    fi
fi

print_success "All done! Happy coding! 🚀"
