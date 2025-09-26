# api/routes/health.py
import os

from fastapi import APIRouter

router = APIRouter()

REQUIRED_ENVS = ["OPENAI_API_KEY", "PINECONE_API_KEY"]


@router.get("/health")
def health():
    # 1) env presence
    envs = {k: bool(os.getenv(k)) for k in REQUIRED_ENVS}

    # 2) filesystem write check (root folder)
    fs_write = False
    try:
        with open(".__sme_write_test", "w") as f:
            f.write("ok")
        os.remove(".__sme_write_test")
        fs_write = True
    except Exception:
        fs_write = False

    # overall
    ok = all(envs.values()) and fs_write
    return {
        "status": "ok" if ok else "degraded",
        "checks": {"env": envs, "fs_write": fs_write},
    }
