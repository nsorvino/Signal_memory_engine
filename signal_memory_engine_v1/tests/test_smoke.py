# tests/test_smoke.py
"""
Wrapper for smoke_test.py to run via pytest.
"""
from scripts.smoke_test import run_smoke


def test_smoke():
    run_smoke(k=3, limit=5)
