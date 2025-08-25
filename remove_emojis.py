#!/usr/bin/env python3
"""
Script to remove emojis from all files in the project
"""

import os
import re
from pathlib import Path

# Common emojis to remove
EMOJI_PATTERN = re.compile(r'[]')

# File extensions to process
EXTENSIONS = {'.py', '.js', '.jsx', '.ts', '.tsx', '.md', '.tex', '.txt', '.yml', '.yaml', '.json'}

def remove_emojis_from_file(file_path):
    """Remove emojis from a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file has emojis
        if EMOJI_PATTERN.search(content):
            # Remove emojis
            new_content = EMOJI_PATTERN.sub('', content)
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"Removed emojis from: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all files"""
    project_root = Path(".")
    files_processed = 0
    files_modified = 0
    
    for root, dirs, files in os.walk(project_root):
        # Skip certain directories
        if any(skip_dir in root for skip_dir in ['node_modules', '.git', '__pycache__', 'venv', 'build', 'dist']):
            continue
            
        for file in files:
            file_path = Path(root) / file
            
            # Check if file has relevant extension
            if file_path.suffix.lower() in EXTENSIONS:
                files_processed += 1
                if remove_emojis_from_file(file_path):
                    files_modified += 1
    
    print(f"\nProcessing complete:")
    print(f"Files processed: {files_processed}")
    print(f"Files modified: {files_modified}")

if __name__ == "__main__":
    main()