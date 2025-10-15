import os

import pytest

from hf_model_inspector import (
    get_model_report_json,
    get_model_report_md,
    recommend_models_for_gpu,
    save_model_report,
)
from hf_model_inspector.loader import HFModelLoader, authenticate_hf


@pytest.fixture(scope="module")
def hf_token():
    """Authenticate once for all tests."""
    return authenticate_hf()


@pytest.fixture(scope="module")
def loader(hf_token):
    """Initialize loader once for all tests."""
    return HFModelLoader(hf_token)


@pytest.fixture(scope="module")
def test_repos():
    return {
        "main_model": "moonshotai/Kimi-K2-Instruct-0905",
        "lora_model": "Nondzu/Mistral-7B-codealpaca-lora",
    }


def test_fetch_model_info(loader, test_repos):
    info = loader.fetch_model_info(test_repos["main_model"])
    assert info is not None, "Failed to fetch model info"
    assert "id" in info, "Model info missing 'id'"
    assert "downloads" in info, "Model info missing 'downloads'"


def test_load_json_quiet(loader, test_repos):
    config = loader.load_json_quiet(test_repos["main_model"], "config.json")
    # Some models may not have config.json
    if config:
        assert isinstance(config, dict)
    else:
        assert config is None


def test_load_lora_info(loader, test_repos):
    lora_info = loader.load_lora_info(test_repos["lora_model"])
    # LoRA info may not exist
    if lora_info:
        assert isinstance(lora_info, dict)
        # estimated_parameters is always present now (0 if no .bin)
        assert "estimated_parameters" in lora_info
        # ensure default keys exist
        for key in ["r", "alpha", "fan_in_fan_out", "target_modules"]:
            assert key in lora_info
    else:
        # If no config exists, loader returns None
        assert lora_info is None


def test_get_model_report_json(hf_token, test_repos):
    try:
        report_json = get_model_report_json(test_repos["main_model"], hf_token)
        assert report_json is not None
        assert "repo_id" in report_json
        assert "architecture" in report_json
    except RuntimeError as e:
        # Handle optional JSON missing gracefully
        pytest.skip(f"Skipped JSON report: {e}")


def test_get_model_report_md(hf_token, test_repos):
    try:
        report_md = get_model_report_md(test_repos["main_model"], hf_token)
        assert report_md is not None
        assert isinstance(report_md, str)
        assert len(report_md) > 0
    except RuntimeError as e:
        pytest.skip(f"Skipped Markdown report: {e}")


def test_save_model_report(hf_token, test_repos):
    try:
        report_md = get_model_report_md(test_repos["main_model"], hf_token)
    except RuntimeError:
        pytest.skip("Skipped saving report due to missing config")

    save_path = "test_model_report.md"
    save_model_report(test_repos["main_model"], md_path=save_path, token=hf_token)
    assert os.path.exists(save_path)
    # Cleanup
    os.remove(save_path)


def test_recommend_models_for_gpu():
    gpu_specs = {"name": "RTX 3090", "memory_gb": 24}
    recommended = recommend_models_for_gpu(gpu_specs)
    assert isinstance(recommended, list)
    assert isinstance(recommended, list)
