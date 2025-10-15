import os
import pytest
from hf_model_inspector import (
    get_model_report_json,
    get_model_report_md,
    save_model_report,
    get_lora_info,
    recommend_models_for_gpu,
)
from hf_model_inspector.loader import authenticate_hf, HFModelLoader


@pytest.fixture(scope="module")
def hf_token():
    # Authenticate once for all tests
    return authenticate_hf()


@pytest.fixture(scope="module")
def loader(hf_token):
    return HFModelLoader(hf_token)


@pytest.fixture(scope="module")
def test_repos():
    return {
        "main_model": "moonshotai/Kimi-K2-Instruct-0905",
        "lora_model": "zai-org/GLM-4.6"
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
        assert "estimated_parameters" in lora_info
    else:
        assert lora_info is None


def test_get_model_report_json(hf_token, test_repos):
    report_json = get_model_report_json(test_repos["main_model"], hf_token)
    assert report_json is not None
    assert "repo_id" in report_json
    assert "architecture" in report_json


def test_get_model_report_md(hf_token, test_repos):
    report_md = get_model_report_md(test_repos["main_model"], hf_token)
    assert report_md is not None
    assert isinstance(report_md, str)
    assert len(report_md) > 0


def test_save_model_report(hf_token, test_repos):
    # Generate markdown and save
    report_md = get_model_report_md(test_repos["main_model"], hf_token)
    save_path = "test_model_report.md"
    save_model_report(test_repos["main_model"], md_path=save_path, token=hf_token)
    assert os.path.exists(save_path)
    # Cleanup
    os.remove(save_path)


def test_recommend_models_for_gpu():
    gpu_specs = {"name": "RTX 3090", "memory_gb": 24}
    recommended = recommend_models_for_gpu(gpu_specs)
    assert isinstance(recommended, list)