"""
Integration tests for invalid/unsupported models.
Tests that the library correctly handles non-LLM models, missing configs, and nonexistent repos.
"""
import pytest

from hf_model_inspector import (
    get_model_report_json,
    get_model_report_md,
)
from hf_model_inspector.loader import HFModelLoader, authenticate_hf


@pytest.fixture(scope="module")
def hf_token():
    """Authenticate once for all integration tests."""
    return authenticate_hf()


@pytest.fixture(scope="module")
def invalid_repos():
    """
    Non-language-model HuggingFace repos.
    These repos exist but don't have standard LLM config.json files.
    """
    return {
        "diffuser_model": "black-forest-labs/FLUX.1-schnell",
        "diffuser_controlnet": "InstantX/Qwen-Image-ControlNet-Union",
        "malformed_repo": "user///bad_model_name",
        "nonexistent_repo": "fake-user-12345/nonexistent-model-99999",
    }


# ---------------------- FETCH INFO TESTS ---------------------- #

def test_fetch_invalid_model_info(hf_token, invalid_repos):
    """
    fetch_model_info returns metadata even for non-LLM models if they exist on HF.
    Only truly nonexistent repos will raise errors.
    """
    loader = HFModelLoader(token=hf_token)
    
    # Diffuser models exist on HF and return metadata (not None)
    diffuser_model = invalid_repos["diffuser_model"]
    result = loader.fetch_model_info(diffuser_model)
    assert result is not None
    assert "id" in result
    assert result["id"] == diffuser_model
    
    # Nonexistent repos should fail
    nonexistent = invalid_repos["nonexistent_repo"]
    with pytest.raises(RuntimeError):
        loader.fetch_model_info(nonexistent)


# ---------------------- REPORT GENERATION TESTS ---------------------- #

def test_get_model_report_json_invalid(hf_token, invalid_repos):
    """
    get_model_report_json should raise ValueError when config.json is missing.
    This is expected and correct behavior for non-LLM models.
    """
    for name, repo_id in invalid_repos.items():
        with pytest.raises((RuntimeError, ValueError)):
            get_model_report_json(repo_id, hf_token)


def test_get_model_report_md_invalid(hf_token, invalid_repos):
    """
    get_model_report_md should raise errors for repos without config.json.
    """
    for name, repo_id in invalid_repos.items():
        with pytest.raises((RuntimeError, ValueError)):
            get_model_report_md(repo_id, hf_token)


# ---------------------- MIXED BATCH TEST ---------------------- #

def test_mixed_batch_with_invalid_and_valid_models(hf_token):
    """
    Mixed batch: valid model should succeed, invalid model should fail gracefully.
    Uses publicly accessible models that don't require gating.
    """
    valid_model = "gpt2"
    invalid_model = "black-forest-labs/FLUX.1-schnell"

    # Valid model should succeed
    valid_result = get_model_report_json(valid_model, hf_token)
    assert valid_result is not None
    assert "repo_id" in valid_result
    assert valid_result["repo_id"] == valid_model

    # Invalid model should raise ValueError due to missing config.json
    with pytest.raises(ValueError):
        get_model_report_json(invalid_model, hf_token)


# ---------------------- BATCH ERROR HANDLING ---------------------- #

def test_batch_with_mixed_results(hf_token):
    """
    Test that a batch of mixed valid/invalid models can be processed with proper error handling.
    """
    models = {
        "valid1": "gpt2",
        "valid2": "distilbert-base-uncased",
        "invalid1": "black-forest-labs/FLUX.1-schnell",
        "invalid2": "nonexistent-user/fake-model-xyz",
    }
    
    results = {}
    errors = {}
    
    for name, repo_id in models.items():
        try:
            result = get_model_report_json(repo_id, hf_token)
            results[name] = result
        except (RuntimeError, ValueError) as e:
            errors[name] = str(e)
    
    # Should have 2 successes and 2 failures
    assert len(results) == 2
    assert len(errors) == 2
    
    # Verify the valid ones succeeded
    assert "valid1" in results
    assert "valid2" in results
    
    # Verify the invalid ones failed
    assert "invalid1" in errors
    assert "invalid2" in errors


# ---------------------- SPECIFIC ERROR MESSAGE TESTS ---------------------- #

def test_config_missing_error_message(hf_token):
    """
    Verify that models without config.json raise ValueError with clear message.
    """
    diffuser_model = "black-forest-labs/FLUX.1-schnell"
    
    with pytest.raises(ValueError) as exc_info:
        get_model_report_json(diffuser_model, hf_token)
    
    assert "Could not load config.json" in str(exc_info.value)
    assert diffuser_model in str(exc_info.value)


def test_nonexistent_model_error_message(hf_token):
    """
    Verify that nonexistent models raise RuntimeError.
    """
    fake_model = "fake-user-12345/nonexistent-model-99999"
    
    with pytest.raises(RuntimeError):
        get_model_report_json(fake_model, hf_token)