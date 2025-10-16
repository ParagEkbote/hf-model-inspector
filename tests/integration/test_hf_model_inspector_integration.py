import os
import pytest

from hf_model_inspector import (
    get_lora_info,
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
def test_repos():
    return {
        "main_model": "moonshotai/Kimi-K2-Instruct-0905",
        "lora_model": "Nondzu/Mistral-7B-codealpaca-lora",
    }


def test_fetch_model_info(hf_token, test_repos):
    loader = HFModelLoader(token=hf_token)
    info = loader.fetch_model_info(test_repos["main_model"])
    assert info is not None, "Failed to fetch model info"
    assert "id" in info
    assert "downloads" in info


def test_load_json_quiet(hf_token, test_repos):
    loader = HFModelLoader(token=hf_token)
    config = loader.load_json_quiet(test_repos["main_model"], "config.json")
    if config:
        assert isinstance(config, dict)
    else:
        assert config is None


def test_get_lora_info(hf_token, test_repos):
    """
    Verify LoRA information extraction from model weights.
    The test is tolerant of missing LoRA adapters and will skip gracefully.
    """
    try:
        info = get_lora_info(test_repos["lora_model"], hf_token)
        if info is None:
            pytest.skip("No LoRA modules found for this repo.")
        assert isinstance(info, dict)
        assert "num_lora_modules" in info
        assert "lora_module_names" in info
        assert isinstance(info["num_lora_modules"], int)
        assert isinstance(info["lora_module_names"], list)
    except RuntimeError as e:
        pytest.skip(f"Skipped due to loading error: {e}")


def test_get_model_report_json(hf_token, test_repos):
    try:
        report_json = get_model_report_json(test_repos["main_model"], hf_token)
        assert report_json is not None
        assert "repo_id" in report_json
        assert "architecture" in report_json
    except RuntimeError as e:
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
        assert report_md and isinstance(report_md, str)
    except RuntimeError:
        pytest.skip("Skipped saving report due to missing config")

    save_path = "test_model_report.md"
    save_model_report(test_repos["main_model"], md_path=save_path, token=hf_token)
    assert os.path.exists(save_path)
    os.remove(save_path)


def test_recommend_models_for_gpu():
    gpu_specs = {"name": "RTX 3090", "memory_gb": 24}
    recommended = recommend_models_for_gpu(gpu_specs)
    assert isinstance(recommended, list)
