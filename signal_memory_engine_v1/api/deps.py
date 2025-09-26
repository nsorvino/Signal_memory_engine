# ============================================================================
# api/deps.py (shared singletons & config init)
# ============================================================================
import logging
import os
import re
from pathlib import Path

import mlflow
from dotenv import load_dotenv

# agents
from agents.axis_agent import ROLE_AXIS, get_axis_chain
from agents.m_agent import ROLE_SENTINEL, get_sentinel_chain
from agents.oria_agent import ROLE_ORIA, get_oria_chain
from coherence.commons import SUGGESTIONS as _SUGGESTIONS
from coherence.commons import flag_from_score as _flag_from_score
from scripts.langchain_retrieval import build_qa_chain
from vector_store.embeddings import get_embedder
from vector_store.pinecone_index import init_pinecone_index

logger = logging.getLogger(__name__)
load_dotenv()

# env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
INDEX_NAME = os.getenv("PINECONE_INDEX", "signal-engine")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not (PINECONE_API_KEY and OPENAI_API_KEY):
    raise RuntimeError("Set PINECONE_API_KEY and OPENAI_API_KEY in .env")

# mlflow basics (file:// fallback, experiment by name)
mlflow.autolog()
try:
    import mlflow.openai

    mlflow.openai.autolog()
except Exception:
    logger.warning("mlflow-openai plugin not installed; skipping OpenAI autolog.")

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)
else:
    project_root = Path(__file__).resolve().parents[1]
    mlruns_dir = project_root / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    try:
        mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    except Exception:
        mlflow.set_tracking_uri(str(mlruns_dir.resolve()))

# EXP_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "signal-memory-engine")
# _client = MlflowClient()
# if _client.get_experiment_by_name(EXP_NAME) is None:
#     _client.create_experiment(EXP_NAME)
# mlflow.set_experiment(EXP_NAME)

# pinecone + embedder + default QA chain
init_pinecone_index(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
    index_name=INDEX_NAME,
    dimension=384,
    metric="cosine",
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

embeddings = get_embedder(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
qa, vectorstore = build_qa_chain(
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_env=PINECONE_ENV,
    index_name=INDEX_NAME,
    openai_api_key=OPENAI_API_KEY,
    embed_model=EMBED_MODEL,
    llm_model=LLM_MODEL,
    k=3,
)


# shared helpers/consts used by routes
def sanitize_key(key: str) -> str:
    # keep alnum, _, -, ., :, /, and space (like your original)
    return re.sub(r"[^\w\-\.:/ ]", "", str(key))


SUGGESTIONS = _SUGGESTIONS

qa_axis, store_axis = get_axis_chain()
qa_oria, store_oria = get_oria_chain()
qa_sentinel, store_sentinel = get_sentinel_chain()

ROLE_CHAINS = (
    (ROLE_AXIS, qa_axis, store_axis),
    (ROLE_ORIA, qa_oria, store_oria),
    (ROLE_SENTINEL, qa_sentinel, store_sentinel),
)

AGENTS = ROLE_CHAINS

flag_from_score = _flag_from_score
