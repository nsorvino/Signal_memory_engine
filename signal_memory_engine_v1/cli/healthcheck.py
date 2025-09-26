# cli/healthcheck.py
import json
import sys
import urllib.request


def main():
    URL = "http://localhost:8000/health"

    try:
        with urllib.request.urlopen(URL, timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
        sys.exit(0 if data.get("status") == "ok" else 1)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
