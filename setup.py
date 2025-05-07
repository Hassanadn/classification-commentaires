# setup.py
from setuptools import setup, find_packages

setup(
    name="classification-commentaires",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "scikit-learn>=1.0.2",
        "pandas>=1.3.3",
        "numpy>=1.21.5",
        "pyyaml>=6.0",
        "joblib>=1.1.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
    ],
)