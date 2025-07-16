# Migration from Conda to Virtual Environment (venv)

## Summary of Changes

This document outlines the migration from conda environment to Python virtual environment (venv) for the Clinical RAG System project.

## Files Modified

### 1. **setup_environment.bat** (Windows Setup Script)
- **Changed**: Now creates Python virtual environment instead of conda environment
- **Removed**: Conda dependency checks and environment creation
- **Added**: Python version verification and venv creation
- **Updated**: Activation commands to use `venv\Scripts\activate.bat`

### 2. **setup_environment.sh** (Linux/macOS Setup Script)  
- **Changed**: Now creates Python virtual environment instead of conda environment
- **Removed**: Conda dependency checks and environment creation
- **Added**: Python version verification and venv creation
- **Updated**: Activation commands to use `source venv/bin/activate`

### 3. **requirements.txt** (Python Dependencies)
- **Updated**: Installation instructions to use venv instead of conda
- **Maintained**: All existing package versions for consistency
- **Added**: Clear step-by-step installation guide

### 4. **README.md** (Project Documentation)
- **Updated**: Installation section to use venv instead of conda
- **Added**: Automated setup script instructions
- **Removed**: References to `langchain_rag_env.yml`
- **Updated**: Project structure to reflect current files

### 5. **dev_config.yml** (Development Configuration)
- **Changed**: `conda_environment: "langchain_rag"` → `virtual_environment: "venv"`
- **Updated**: Environment references throughout the file

### 6. **.gitignore** (Version Control Exclusions)
- **Removed**: Conda-specific exclusions (`.conda/`, `conda-meta/`)
- **Added**: Virtual environment activation scripts to exclusions
- **Maintained**: All other exclusion patterns

## Files Deleted

### 1. **langchain_rag_env.yml** (Conda Environment File)
- **Reason**: No longer needed as project uses venv
- **Replacement**: Standard `requirements.txt` with pip installation

## New Files Created

### 1. **requirements-dev.txt** (Development Dependencies)
- **Purpose**: Separate development tools and testing dependencies
- **Contents**: Black, flake8, mypy, pytest, Jupyter, documentation tools
- **Usage**: `pip install -r requirements-dev.txt` for development setup

## Migration Benefits

### **Simplified Setup**
- No conda dependency - works with any Python 3.11+ installation
- Faster environment creation and activation
- Standard Python virtual environment approach

### **Better Portability**
- Works on any system with Python installed
- No need for Anaconda/Miniconda installation
- Consistent across different operating systems

### **Reduced Complexity**
- Single requirements.txt file for all dependencies
- Eliminates conda/pip mixing issues
- Clearer dependency management

## Usage Instructions

### For New Users
```bash
# Clone repository
git clone <repository-url>
cd msc_project

# Use automated setup (recommended)
# Windows:
setup_environment.bat

# Linux/macOS:
chmod +x setup_environment.sh
./setup_environment.sh
```

### Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For development:
pip install -r requirements-dev.txt
```

### Environment Activation
```bash
# Windows:
venv\Scripts\activate.bat

# Linux/macOS:
source venv/bin/activate

# Deactivate (any platform):
deactivate
```

## Verification

After migration, verify the setup works:

```bash
# Activate environment
source venv/bin/activate  # or venv\Scripts\activate.bat on Windows

# Test key imports
python -c "
import langchain
import faiss
import sentence_transformers
import pandas
import torch
print('✅ All packages imported successfully!')
"

# Run a quick test
cd RAG_chat_pipeline
python rag_evaluator.py quick
```

## Backward Compatibility

- **Requirements**: Python 3.11+ (previously handled by conda)
- **Dependencies**: All package versions maintained for consistency
- **Functionality**: No changes to core RAG system functionality
- **Data**: All existing data files and vector stores remain compatible

## Notes

1. **Development Tools**: Now available in separate `requirements-dev.txt`
2. **Automated Scripts**: Both Windows and Linux/macOS setup scripts updated
3. **Documentation**: README.md updated with new installation instructions
4. **Git Tracking**: Activation scripts added to .gitignore
5. **Environment Name**: Changed from `langchain_rag` to `venv` (standard)

This migration simplifies the setup process while maintaining all existing functionality and dependencies.
