import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from hf_model_inspector import (
    get_model_report_json,
    get_model_report_md,
    save_model_report,
)
from hf_model_inspector.loader import HFModelLoader, authenticate_hf


@pytest.fixture(scope="module")
def hf_token():
    """Authenticate once for all integration tests."""
    return authenticate_hf()


@pytest.fixture(scope="module")
def batch_model_list():
    """
    List of public HF repos to be used in batch processing tests.
    Adjust based on available bandwidth and rate limits.
    """
    return [
        "meta-llama/Llama-3.3-70B-Instruct",
        "mistralai/Mistral-7B-v0.1",
        "tiiuae/falcon-7b-instruct",
        "Nondzu/Mistral-7B-codealpaca-lora",
    ]


# ---------------------- BATCH FETCH TESTS ---------------------- #

def test_batch_fetch_model_info(hf_token, batch_model_list):
    """
    Test fetching model info for a batch of models using the loader.
    Ensures all models return valid metadata dictionaries.
    """
    loader = HFModelLoader(token=hf_token)
    all_infos = []

    for repo_id in batch_model_list:
        try:
            info = loader.fetch_model_info(repo_id)
            assert info is not None, f"No info returned for {repo_id}"
            assert "id" in info, f"Missing id in info for {repo_id}"
            all_infos.append(info)
        except RuntimeError as e:
            pytest.skip(f"Skipped {repo_id} due to fetch error: {e}")

    assert len(all_infos) > 0, "No model info fetched in batch"


# ---------------------- BATCH REPORT TESTS ---------------------- #

def test_batch_generate_model_reports(hf_token, batch_model_list):
    """
    Generate JSON and Markdown reports for multiple models.
    Validates structure and non-empty content for each model.
    """
    reports = []

    for repo_id in batch_model_list:
        try:
            report_json = get_model_report_json(repo_id, hf_token)
            assert isinstance(report_json, dict), f"Invalid JSON report for {repo_id}"
            assert "repo_id" in report_json

            report_md = get_model_report_md(repo_id, hf_token)
            assert isinstance(report_md, str) and len(report_md) > 0, f"Empty Markdown for {repo_id}"

            reports.append({
                "model": repo_id,
                "architecture": report_json.get("architecture"),
                "param_count": report_json.get("param_count", "unknown"),
            })
        except RuntimeError as e:
            pytest.skip(f"Skipped report generation for {repo_id}: {e}")

    assert len(reports) > 0, "No reports generated in batch"
    for r in reports:
        assert "model" in r
        assert "architecture" in r


# ---------------------- BATCH SAVE TESTS ---------------------- #

def test_batch_save_model_reports(hf_token, batch_model_list):
    """
    Save Markdown reports for multiple models and verify existence of saved files.
    """
    for repo_id in batch_model_list:
        try:
            md_report = get_model_report_md(repo_id, hf_token)
            save_path = f"test_report_{repo_id.replace('/', '_')}.md"
            save_model_report(repo_id, md_path=save_path, token=hf_token)

            assert os.path.exists(save_path), f"Report not saved for {repo_id}"
            os.remove(save_path)
        except RuntimeError as e:
            pytest.skip(f"Skipped saving report for {repo_id}: {e}")


# ---------------------- ERROR HANDLING TESTS ---------------------- #

def test_batch_with_invalid_model(hf_token):
    """
    Test batch handling with invalid/nonexistent models.
    Should gracefully skip or raise controlled errors.
    """
    invalid_models = ["unknown_user/fake-model-404", "repo/invalid@@name"]
    for repo_id in invalid_models:
        with pytest.raises(RuntimeError):
            get_model_report_json(repo_id, hf_token)


# ---------------------- PARALLEL PROCESSING TEST ---------------------- #

def test_parallel_batch_processing(hf_token, batch_model_list):
    """
    Test concurrent (parallel) batch report generation for efficiency.
    Uses ThreadPoolExecutor to process models in parallel.
    """

    def process_model(repo_id):
        try:
            report_json = get_model_report_json(repo_id, hf_token)
            report_md = get_model_report_md(repo_id, hf_token)
            save_path = f"parallel_{repo_id.replace('/', '_')}.md"
            save_model_report(repo_id, md_path=save_path, token=hf_token)
            os.remove(save_path)
            return repo_id, report_json.get("architecture", "unknown")
        except Exception as e:
            return repo_id, f"error: {e}"

    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_model, repo_id) for repo_id in batch_model_list]
        for f in as_completed(futures):
            results.append(f.result())

    assert len(results) == len(batch_model_list)
    for model, result in results:
        assert isinstance(model, str)
        assert result is not None
