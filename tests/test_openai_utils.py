from pathlib import Path

from llama_index.core import Settings

from rag_service.config import AppConfig
from rag_service.openai_utils import init_llamaindex_clients


def test_init_warns_on_no_ssl(monkeypatch, caplog):
    """OpenAI client initialization should warn when SSL verification is off."""

    cfg = AppConfig.load(Path("config.json"))
    cfg.openai.embeddings.verify_ssl = False
    cfg.openai.generator.verify_ssl = False
    monkeypatch.setenv("OPENAI_API_KEY_EMB", "e")
    monkeypatch.setenv("OPENAI_API_KEY_GEN", "g")
    with caplog.at_level("WARNING"):
        init_llamaindex_clients(cfg)
        assert "SSL verification disabled" in caplog.text
    assert Settings.llm is not None
    assert Settings.embed_model is not None
