# src/__init__.py
# This can be empty, it just marks the directory as a Python package

# setup.py
from setuptools import setup, find_packages

setup(
    name="skin-lesion-analyzer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "Pillow",
        "opencv-python",
        "scikit-learn",
        "lightgbm",
    ],
)