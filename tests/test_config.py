import os

from src import config


def test_offline_env_vars_are_forced():
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_paths_are_resolved_under_project_root():
    assert str(config.DATA_NORMAS_DIR).startswith(str(config.PROJECT_ROOT))
    assert str(config.MODELS_DIR).startswith(str(config.PROJECT_ROOT))
    assert str(config.CHROMA_PERSIST_DIR).startswith(str(config.PROJECT_ROOT))


def test_veredito_values_match_specification():
    assert config.VEREDITOS == (
        "CONFORME",
        "NAO_CONFORME",
        "NAO_APLICAVEL",
        "INDETERMINADO",
    )


def test_retrieval_strategies_are_the_four_experiment_configs():
    assert config.RETRIEVAL_STRATEGIES == ("A", "B", "C", "D")
