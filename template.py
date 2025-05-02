# Library Installation
import os
from pathlib import Path
import logging

# Logging Function
logging.basicConfig(level=logging.INFO, format='[%(astime)s]: %(message)s:')

# Python List
list_of_files = [
    "src/__init__.py", # Constructor File
    "src/helper.py", # Consists all the functionalities
    "src/prompt.py", # System Prompts
    ".env",
    "setup.py", # Local Package Folder Installation
    "app.py", 
    "research/trials.ipynb" # Jupyter Notebook Experiments
]

for filepath in list_of_files: # Iterating over the List of Files
    filepath = Path(filepath) # For converting file paths (Multiple OS Operation)
    filedir, filename = os.path.split(filepath) # For Splitting Folder & File names
    
    
    if filedir !="": # Checking if the Folder exisits
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}") # Log message
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w")as f: # File creation
            pass
            logging.info(f"Creating empty file: {filepath}")
            
    else:
        logging.info(f"{filename} is already exists") # File aleeady exists