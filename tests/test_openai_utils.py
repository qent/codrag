from pathlib import Path

from llama_index.core import Settings

from rag_service.config import AppConfig
import rag_service.openai_utils as ou


def test_init_warns_on_no_ssl(monkeypatch, caplog):
    """OpenAI client initialization should warn when SSL verification is off."""

    cfg = AppConfig.load(Path("config.json"))
    cfg.openai.embeddings.verify_ssl = False
    cfg.openai.generator.verify_ssl = False
    monkeypatch.setenv("OPENAI_API_KEY_EMB", "e")
    monkeypatch.setenv("OPENAI_API_KEY_GEN", "g")
    with caplog.at_level("WARNING"):
        ou.init_llamaindex_clients(cfg)
        assert "SSL verification disabled" in caplog.text
    assert Settings.llm is not None
    assert Settings.embed_model is not None
    ou.close_llamaindex_clients()


def test_clients_stored_and_closed(monkeypatch):
    """init_llamaindex_clients should expose and close HTTP clients."""

    cfg = AppConfig.load(Path("config.json"))
    monkeypatch.setenv("OPENAI_API_KEY_EMB", "e")
    monkeypatch.setenv("OPENAI_API_KEY_GEN", "g")
    ou.init_llamaindex_clients(cfg)
    assert ou.EMBEDDINGS_CLIENT is not None
    assert ou.GENERATOR_CLIENT is not None
    emb = ou.EMBEDDINGS_CLIENT
    gen = ou.GENERATOR_CLIENT
    ou.close_llamaindex_clients()
    assert emb.is_closed
    assert gen.is_closed
    assert ou.EMBEDDINGS_CLIENT is None
    assert ou.GENERATOR_CLIENT is None
