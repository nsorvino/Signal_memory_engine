from setuptools import find_packages, setup

setup(
    name="signal_memory_engine_v1",
    version="2.0.0",
    packages=find_packages(exclude=("tests", "scripts")),
    python_requires=">=3.10",
    install_requires=[
        "fastapi~=0.116",
        "uvicorn~=0.35",
        "httpx~=0.28",
        "pydantic~=2.11",
        "openai~=1.106",
        "langchain~=0.3",
        "langchain-community~=0.3",
        "langchain-openai~=0.3",
        "langchain-pinecone~=0.2",
        "pinecone~=7.3",
        "sentence-transformers~=5.1",
        "mlflow~=3.2",
    ],
    extras_require={
        # Optional UI
        "ui": [
            "streamlit~=1.49",
        ],
        # Local dev & CI tools
        "dev": [
            "pytest~=8.4",
            "pytest-cov~=5.0",
            "ruff~=0.13",
            "mypy~=1.11",
            "types-requests~=2.32",
        ],
        # Optional docs
        "docs": [
            "mkdocs-material~=9.5",
        ],
    },
    include_package_data=True,
)
