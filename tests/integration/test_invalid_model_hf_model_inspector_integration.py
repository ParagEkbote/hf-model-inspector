import pytest
from hf_model_inspector import get_model_report_json, get_model_report_md
from hf_model_inspector.loader import HFModelLoader, authenticate_hf


@pytest.fixture(scope="module")
def hf_token():
    """Authenticate once for all invalid model tests."""
    return authenticate_hf()


@pytest.fixture(scope="module")
def invalid_repos():
    """
    Repositories expected to be unsupported for model inspection:
    diffusers pipelines, datasets, spaces, malformed paths, non-existent repos.
    """
    return {
        "diffuser_model": "black-forest-labs/FLUX.1-schnell",
        "diffuser_controlnet": "InstantX/Qwen-Image-ControlNet-Union",
        "dataset_repo": "datasets/fka/awesome-chatgpt-prompts",
        "space_repo": "spaces/tencent/Hunyuan3D-2.1",
        "malformed_repo": "user///bad_model_name",
        "nonexistent_repo": "fakeuser/some-random-model-that-does-not-exist",
    }


# ---------------------- BASIC INVALID MODEL TESTS ---------------------- #

def test_fetch_invalid_model_info(hf_token, invalid_repos):
    """Fetching invalid models should return None."""
    loader = HFModelLoader(token=hf_token)
    for name, repo_id in invalid_repos.items():
        result = loader.fetch_model_info(repo_id)
        assert result is None, f"{repo_id} should return None for unsupported models"


# ---------------------- JSON REPORT FAILURE TEST ---------------------- #

def test_get_model_report_json_invalid(hf_token, invalid_repos):
    """get_model_report_json should return None for unsupported repos."""
    for name, repo_id in invalid_repos.items():
        result = get_model_report_json(repo_id, hf_token)
        assert result is None, f"{repo_id} should return None for unsupported models"


# ---------------------- MARKDOWN REPORT FAILURE TEST ---------------------- #

def test_get_model_report_md_invalid(hf_token, invalid_repos):
    """get_model_report_md should return None or contain 'unsupported' for unsupported repos."""
    for name, repo_id in invalid_repos.items():
        result = get_model_report_md(repo_id, hf_token)
        assert result is None or "unsupported" in result.lower(), (
            f"{repo_id} should be unsupported, got: {result}"
        )


# ---------------------- MIXED BATCH INVALID HANDLING ---------------------- #

def test_mixed_batch_with_invalid_and_valid_models(hf_token):
    """
    Mixed batch: valid model should succeed, invalid model should return None or unsupported.
    """
    valid_model = "meta-llama/Llama-2-7b-chat-hf"
    invalid_model = "runwayml/stable-diffusion-v1-5"

    # Valid model should succeed
    valid_result = get_model_report_json(valid_model, hf_token)
    assert valid_result and "repo_id" in valid_result

    # Invalid model should return None or indicate unsupported
    invalid_result = get_model_report_json(invalid_model, hf_token)
    assert invalid_result is None or "unsupported" in str(invalid_result).lower()
