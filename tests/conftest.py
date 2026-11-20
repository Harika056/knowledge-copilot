import pytest

from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.llm import OfflineLLM
from copilot.pipeline import Copilot
from copilot.sample import build_sample_db
from copilot.store import MemoryStore

DOCS = "sample_data/docs"


@pytest.fixture()
def settings(tmp_path):
    db = tmp_path / "sales.db"
    build_sample_db(str(db))
    return Settings(jwt_secret="test-secret", sql_db_path=str(db))


@pytest.fixture()
def copilot(settings):
    cp = Copilot(MemoryStore(), HashEmbedder(), OfflineLLM(), settings)
    cp.ingest_folder(DOCS)
    return cp
