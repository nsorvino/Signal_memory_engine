# ============================================================================
# api/main.py (slim bootstrap)
# ============================================================================
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import deps
from api.routes import agents as agents_routes
from api.routes import health as health_routes
from api.routes import memory as memory_routes
from api.routes import multi as multi_routes
from api.routes import query as query_routes
from api.routes import score as score_routes
from api.routes import signal as signal_routes
from utils.logging_setup import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # force construction once at startup (touch deps)
    _ = (deps.qa, deps.vectorstore, deps.AGENTS)
    yield


app = FastAPI(title="Signal Memory RAG API", lifespan=lifespan)

# routers
app.include_router(signal_routes.router)
app.include_router(query_routes.router)
app.include_router(multi_routes.router)
app.include_router(memory_routes.router)
app.include_router(agents_routes.router)
app.include_router(score_routes.router)
app.include_router(health_routes.router)
