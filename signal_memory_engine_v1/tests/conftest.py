# signal_memory_engine_v1/tests/conftest.py
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# ---------- Load .env and set CI/Test env defaults ----------
load_dotenv()

os.environ.setdefault("SME_TEST_MODE", "1")
os.environ.setdefault("OPENAI_API_KEY", "test-openai")
# Force offline path to guarantee stub usage during tests
os.environ.setdefault("PINECONE_API_KEY", "dummy")

# Make sure every index referenced exists in the stub
os.environ.setdefault("PINECONE_INDEX", "axis-memory")
os.environ.setdefault(
    "PINECONE_INDEXES",
    "axis-memory,sentinel-memory,oria-memory,test-index,signal-engine",
)
os.environ.setdefault("PINECONE_ENVIRONMENT", "us-east-1")
os.environ.setdefault("PINECONE_DIMENSION", "384")

# ---------- Ensure project root on sys.path ----------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- Install the Pinecone stub (idempotent) ----------
from utils.pinecone_stub import install as _install_pinecone_stub  # noqa: E402

# Always install for tests (safe + deterministic)
_install_pinecone_stub()


# Safety net in case something reloads/overwrites modules during test run
@pytest.fixture(autouse=True)
def _stabilize():
    _install_pinecone_stub(force=True)
    yield
