# Importing required libraries for file and directory operations
import os
from pathlib import Path
import logging

# Defining logging format and level for tracking operations
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Defining the list of essential project files and folders to be created
list_of_files = [
    "src/__init__.py",           # Package initializer file
    "src/helper.py",             # Will contain all helper functions (e.g., PDF loader, embeddings)
    "src/prompt.py",             # Will contain system prompt text
    ".env",                      # Will store API keys and environment variables
    "setup.py",                  # Used for local pip installation (if needed)
    "app.py",                    # Main Flask application
    "research/trials.ipynb"      # Notebook for experimentation and testing
]

# Creating required directories and files based on the above list
for filepath in list_of_files:
    filepath = Path(filepath)                        # Convert to a cross-platform path object
    filedir, filename = os.path.split(filepath)      # Split into directory and file name

    # Creating the directory if it doesn't already exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Creating the file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Just create an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")  # Logging if file already exists
