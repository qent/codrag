from pathlib import Path

from rag_service.config import AppConfig


def test_load_config(tmp_path):
    """Configuration should load minimal JSON correctly."""

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        '{'
        '"version":1,'
        '"indexing":{},'
        '"ast":{},'
        '"openai":{"embeddings":{"base_url":"","model":"m","api_key":"k"},'
        '"generator":{"base_url":"","model":"m","api_key":"k"},'
        '"query_rewriter":{"base_url":"","model":"m","api_key":"k"}},'
        '"prompts":{"file_card_md":"a","dir_card_md":"b"},'
        '"qdrant":{},'
        '"llamaindex":{}'
        '}'
    )
    cfg = AppConfig.load(cfg_path)
    assert cfg.version == 1
    assert cfg.openai.embeddings.model == "m"
    assert cfg.prompts.file_card_md == "a"
