from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hf_model_inspector import (
    get_lora_info,
    get_model_report_json,
    get_model_report_md,
    recommend_models_for_gpu,
    save_model_report,
)


@pytest.fixture
def mock_hf_api():
    """Mock HfApi for model info retrieval."""
    mock_api = MagicMock()

    # Create a mock model info object with attributes
    mock_info = MagicMock()
    mock_info.modelId = "user/test-model"
    mock_info.sha = "abc123"
    mock_info.downloads = 42
    mock_info.likes = 7
    mock_info.tags = ["tagA"]
    mock_info.pipeline_tag = "text-generation"
    mock_info.library_name = "transformers"
    mock_info.private = False
    mock_info.gated = False
    mock_info.author = "user"

    # Mock siblings
    sibling1 = MagicMock()
    sibling1.rfilename = "config.json"
    sibling2 = MagicMock()
    sibling2.rfilename = "tokenizer_config.json"
    mock_info.siblings = [sibling1, sibling2]

    mock_info.cardData = None
    mock_info.lastModified = None
    mock_info.createdAt = None

    mock_api.model_info.return_value = mock_info
    return mock_api


@pytest.fixture
def mock_hf_hub_download():
    """Mock hf_hub_download to return fake file paths."""

    def fake_download(repo_id: str, filename: str, token=None):
        return f"/fake/path/{filename}"

    return fake_download


@pytest.fixture
def mock_json_files():
    """Mock JSON file contents."""

    def fake_open(path, *args, **kwargs):
        if "config.json" in path:
            content = '{"architectures": ["BertModel"], "model_type": "bert"}'
        elif "tokenizer_config.json" in path:
            content = '{"vocab_size": 30000}'
        else:
            content = '{}'

        from io import StringIO

        return StringIO(content)

    return fake_open


def test_get_model_report_json_success(mock_hf_api, mock_hf_hub_download, mock_json_files):
    """Test successful model report generation."""

    with (
        patch("hf_model_inspector.HfApi", return_value=mock_hf_api),
        patch("hf_model_inspector.authenticate_hf", return_value=None),
        patch("hf_model_inspector.hf_hub_download", mock_hf_hub_download),
        patch("builtins.open", mock_json_files),
        patch("hf_model_inspector.estimate_param_count", return_value=(100_000_000, "heuristic")),
        patch("hf_model_inspector.detect_quant_and_precision", return_value={"dtype": "float16"}),
        patch("hf_model_inspector.analyze_tokenizer", return_value={"vocab": 30000}),
        patch("hf_model_inspector.extract_architecture_extras", return_value={"layers": 12}),
    ):

        report = get_model_report_json("user/test-model")
        assert report["repo_id"] == "user/test-model"
        assert report["architecture"] == "BertModel"
        assert report["model_type"] == "bert"
        assert report["parameters"]["total"] == 100_000_000
        assert report["quantization"]["dtype"] == "float16"
        assert report["tokenizer"]["vocab"] == 30000
        assert report["architecture_extras"]["layers"] == 12
        assert report["metadata"]["downloads"] == 42


def test_get_model_report_json_missing_config(mock_hf_api):
    """Should raise ValueError when config.json not found."""

    # Mock API with no config.json in siblings
    mock_info = MagicMock()
    mock_info.modelId = "repo/missing-config"
    mock_info.downloads = 0
    mock_info.likes = 0
    mock_info.tags = []
    mock_info.pipeline_tag = None
    mock_info.library_name = None
    mock_info.private = False
    mock_info.gated = False
    mock_info.author = None
    mock_info.siblings = []
    mock_info.cardData = None
    mock_info.lastModified = None
    mock_info.createdAt = None

    mock_api_no_config = MagicMock()
    mock_api_no_config.model_info.return_value = mock_info

    with (
        patch("hf_model_inspector.HfApi", return_value=mock_api_no_config),
        patch("hf_model_inspector.authenticate_hf", return_value=None),
        patch("hf_model_inspector.hf_hub_download", side_effect=Exception("File not found")),
    ):

        with pytest.raises(ValueError):
            get_model_report_json("repo/missing-config")


def test_get_model_report_md():
    """Ensure Markdown formatting is applied to JSON report."""

    mock_report = {"repo_id": "repo/x"}

    with (
        patch("hf_model_inspector.get_model_report_json", return_value=mock_report),
        patch("hf_model_inspector.format_markdown", return_value="# Report for repo/x"),
    ):

        md = get_model_report_md("repo/x")
        assert md.startswith("# Report for repo/x")


def test_save_model_report():
    """Check that markdown saving is called correctly."""

    with (
        patch(
            "hf_model_inspector.get_model_report_md", return_value="# markdown content"
        ) as mock_get_md,
        patch(
            "hf_model_inspector.save_outputs", return_value="user_modelX_report.md"
        ) as mock_save,
    ):

        result = save_model_report("user/modelX")

        mock_get_md.assert_called_once()
        mock_save.assert_called_once()

        # Check the call arguments
        call_args = mock_save.call_args
        assert call_args[0][0].startswith("# markdown")
        assert call_args[0][1].endswith("_report.md")


def test_get_lora_info_detects_lora(mock_hf_api, mock_hf_hub_download):
    """Should detect LoRA parameter names and return summary."""

    def mock_open_lora(path, *args, **kwargs):
        content = '{"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]}'
        from io import StringIO

        return StringIO(content)

    # Mock load_model_and_tokenizer to return model with LoRA params
    def mock_load_model_tokenizer(repo_id: str):
        param = SimpleNamespace(dtype="float32")
        model = SimpleNamespace(
            named_parameters=lambda: [("loraA.weight", param), ("linear.bias", param)]
        )
        return model, None

    with (
        patch("hf_model_inspector.HfApi", return_value=mock_hf_api),
        patch("hf_model_inspector.authenticate_hf", return_value=None),
        patch("hf_model_inspector.hf_hub_download", mock_hf_hub_download),
        patch("builtins.open", mock_open_lora),
        patch("hf_model_inspector.load_model_and_tokenizer", mock_load_model_tokenizer),
    ):

        info = get_lora_info("repo/has-lora")
        assert info["num_lora_modules"] == 1
        assert "loraA.weight" in info["lora_module_names"]


def test_get_lora_info_none(mock_hf_api, mock_hf_hub_download):
    """If no LoRA params exist, return None."""

    def mock_open_lora(path, *args, **kwargs):
        content = '{"r": 8, "alpha": 16, "target_modules": ["q_proj"]}'
        from io import StringIO

        return StringIO(content)

    # Mock load_model_and_tokenizer to return model WITHOUT LoRA params
    def mock_load_model_tokenizer(repo_id: str):
        param = SimpleNamespace(dtype="float32")
        model = SimpleNamespace(named_parameters=lambda: [("linear.weight", param)])
        return model, None

    with (
        patch("hf_model_inspector.HfApi", return_value=mock_hf_api),
        patch("hf_model_inspector.authenticate_hf", return_value=None),
        patch("hf_model_inspector.hf_hub_download", mock_hf_hub_download),
        patch("builtins.open", mock_open_lora),
        patch("hf_model_inspector.load_model_and_tokenizer", mock_load_model_tokenizer),
    ):

        assert get_lora_info("repo/no-lora") is None


@pytest.mark.parametrize(
    "gpu_specs,expected",
    [
        ({"name": "A100", "memory_gb": 40, "compute_capability": 8.0}, ["large", "medium"]),
        ({"name": "RTX 3080", "memory_gb": 10, "compute_capability": 8.0}, ["medium", "small"]),
        ({"name": "GTX 1080", "memory_gb": 8, "compute_capability": 6.1}, ["small"]),
        ({"name": "Unknown", "memory_gb": 16, "compute_capability": 7.5}, ["medium"]),
    ],
)
def test_recommend_models_for_gpu(gpu_specs: dict[str, float | int | str], expected: list[str]):
    """Test GPU recommendations based on specs."""
    result = recommend_models_for_gpu(gpu_specs)
    assert sorted(result) == sorted(expected)
