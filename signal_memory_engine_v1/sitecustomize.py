# signal_memory_engine_v1/sitecustomize.py
import os

# Auto-install stub when running in test/smoke modes, or using the dummy key
if (
    os.getenv("SME_TEST_MODE") == "1"
    or os.getenv("RUN_API_SMOKE") == "0"
    or os.getenv("PINECONE_API_KEY") == "dummy"
):
    try:
        from utils.pinecone_stub import install as _install_pinecone_stub

        _install_pinecone_stub()
    except Exception:
        # Never block app start if the stub can't be installed
        pass
