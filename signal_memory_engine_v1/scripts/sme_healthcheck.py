#!/usr/bin/env python3
import json
import os
import sys
import urllib.request


def main():
    url = os.environ.get("SME_HEALTH_URL", "http://localhost:8000/health")
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
        ok = data.get("status") == "ok"
    except Exception:
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
