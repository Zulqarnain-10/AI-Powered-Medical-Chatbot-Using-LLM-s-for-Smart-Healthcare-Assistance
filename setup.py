# Importing necessary functions from setuptools for packaging
from setuptools import find_packages, setup

# Defining the setup configuration for the Generative AI project
setup(
    name='Generative AI Project',           # Name of the project/package
    version='0.0.0',                        # Initial version
    author='Syed Zulqarnain Hassan',        # Author information
    author_email='s.zulqarnain1000@gmail.com',
    packages=find_packages(),               # Automatically find all Python packages (with __init__.py)
    install_requires=[]                     # Dependencies can be added here if needed (e.g. Flask, langchain)
)
