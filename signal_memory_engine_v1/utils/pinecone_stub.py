# signal_memory_engine_v1/utils/pinecone_stub.py
from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

__all__ = ["install"]


# --- Fake Pinecone primitives -------------------------------------------------
class Index:
    """Fake pinecone.Index with minimal surface (upsert/query/fetch/stats)."""

    def __init__(self, *a, **k) -> None:
        pass

    def upsert(self, *a, **k):
        return {"upserted": 0}

    def query(self, *a, **k):
        return {"matches": []}

    def fetch(self, *a, **k):
        return {"vectors": {}}

    def describe_index_stats(self, *a, **k):
        return {}


class _IdxObj:
    def __init__(self, name: str) -> None:
        self.name = name


class _IndexList:
    """Compat for both .names() and .index_list['indexes'] usage."""

    def __init__(self, names_list: list[str]) -> None:
        self._names = list(names_list)
        self.index_list = {"indexes": [_IdxObj(n) for n in names_list]}

    def names(self) -> list[str]:
        return list(self._names)


class _IndexDesc:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension


def _env_index_names() -> list[str]:
    raw = os.environ.get("PINECONE_INDEXES")
    if raw:
        return [n.strip() for n in raw.split(",") if n.strip()]
    return [os.environ.get("PINECONE_INDEX", "axis-memory")]


class _PineconeClient:
    def __init__(self, *a, **k) -> None:
        dim = int(os.environ.get("PINECONE_DIMENSION", "384"))
        self._indexes: dict[str, int] = {name: dim for name in _env_index_names()}

    def list_indexes(self) -> _IndexList:
        return _IndexList(list(self._indexes.keys()))

    def describe_index(self, name: str) -> _IndexDesc:
        dim = self._indexes.get(name, int(os.environ.get("PINECONE_DIMENSION", "384")))
        return _IndexDesc(dim)

    def create_index(self, name: str, dimension: int, **kwargs) -> None:
        self._indexes[name] = int(dimension)

    def delete_index(self, name: str) -> None:
        self._indexes.pop(name, None)

    def Index(self, name: str) -> Index:
        return Index()


class _ServerlessSpec:
    def __init__(self, cloud: str = "aws", region: str = "us-east-1") -> None:
        self.cloud = cloud
        self.region = region


# mypy-friendly typed shim for the dynamic module
if TYPE_CHECKING:

    class _PineconeModule(ModuleType):
        Pinecone: type
        ServerlessSpec: type
        Index: type
        exceptions: Any

    _fake_pinecone = cast("_PineconeModule", ModuleType("pinecone"))
else:
    _fake_pinecone = ModuleType("pinecone")

# Idempotent install guard
_installed = False


def install(force: bool = False) -> None:
    """
    Register a fake 'pinecone' module (and legacy submodules) in sys.modules.
    Also ensures vector_store.pinecone_index.index exists for direct imports.
    """
    global _installed

    if _installed and not force:
        return

    # expose top-level API
    setattr(_fake_pinecone, "Pinecone", _PineconeClient)
    setattr(_fake_pinecone, "ServerlessSpec", _ServerlessSpec)
    setattr(_fake_pinecone, "Index", Index)
    setattr(
        _fake_pinecone,
        "exceptions",
        SimpleNamespace(
            exceptions=SimpleNamespace(
                UnauthorizedException=type("UnauthorizedException", (Exception,), {})
            )
        ),
    )

    # legacy path: from pinecone.db_data.index import Index
    db_data_mod = ModuleType("pinecone.db_data")
    db_index_mod = ModuleType("pinecone.db_data.index")
    setattr(db_index_mod, "Index", Index)

    sys.modules["pinecone"] = _fake_pinecone
    sys.modules["pinecone.db_data"] = db_data_mod
    sys.modules["pinecone.db_data.index"] = db_index_mod

    # ensure module exposes `index` when imported directly
    try:
        pci = importlib.import_module("vector_store.pinecone_index")
        if not hasattr(pci, "index"):
            setattr(pci, "index", Index())
    except Exception:
        pass

    _installed = True
