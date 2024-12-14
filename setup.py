from setuptools import setup, find_packages

setup(
    name="lesion-app",
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
        "joblib",
        "matplotlib",
        "scikit-image",
    ],
)