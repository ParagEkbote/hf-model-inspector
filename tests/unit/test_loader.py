import json
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

import hf_model_inspector.loader as loader_mod
from hf_model_inspector.loader import HFModelLoader, authenticate_hf


# -------------------------
# Helpers / fixtures
# -------------------------
@pytest.fixture(autouse=True)
def isolate_auth(monkeypatch):
    """
    Prevent real authentication & subprocess calls during tests by default.
    Tests can override behavior as needed.
    """
    # Default: cached token None, so authenticate_hf would call CLI if not patched.
    monkeypatch.setattr(loader_mod.HfFolder, "get_token", lambda: None)
    yield


# -------------------------
# authenticate_hf tests
# -------------------------
def test_authenticate_hf_token_provided():
    tok = "  my-secret-token  "
    assert authenticate_hf(tok) == "my-secret-token"


def test_authenticate_hf_uses_cached_token(monkeypatch):
    # Simulate a cached token and a successful whoami
    monkeypatch.setattr(loader_mod.HfFolder, "get_token", lambda: "cached-tok")
    monkeypatch.setattr(loader_mod, "whoami", lambda token: {"name": "me"})
    returned = authenticate_hf()
    assert returned == "cached-tok"


def test_authenticate_hf_cli_missing(monkeypatch):
    # No cached token; subprocess.run raises FileNotFoundError
    monkeypatch.setattr(loader_mod.HfFolder, "get_token", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(RuntimeError) as exc:
        authenticate_hf()
    assert "Hugging Face CLI not found" in str(exc.value)


# -------------------------
# fetch_model_info tests
# -------------------------
def test_fetch_model_info_success(monkeypatch):
    # Prevent authenticate_hf from launching CLI
    monkeypatch.setattr(loader_mod, "authenticate_hf", lambda token=None: "tok")
    # Create a fake info object returned by HfApi.model_info
    fake_info = SimpleNamespace(
        modelId="user/model",
        sha="deadbeef",
        downloads=123,
        likes=4,
        tags=["tag1"],
        pipeline_tag=None,
        library_name="transformers",
        private=False,
        gated=False,
        author="author-name",
        siblings=[SimpleNamespace(rfilename="file1.bin")],
        cardData={"desc": "x"},
        lastModified="2024-01-01",
        createdAt="2023-01-01",
    )

    class FakeApi:
        def __init__(self, token=None):
            pass

        def model_info(self, repo_id, token=None):
            return fake_info

    monkeypatch.setattr(loader_mod, "HfApi", FakeApi)
    loader = HFModelLoader(token=None)
    info = loader.fetch_model_info("user/model")
    assert info["id"] == "user/model"
    assert info["downloads"] == 123
    assert info["siblings"] == ["file1.bin"]


def test_fetch_model_info_diffusers_rejected(monkeypatch):
    monkeypatch.setattr(loader_mod, "authenticate_hf", lambda token=None: "tok")
    fake_info = SimpleNamespace(pipeline_tag="diffusers")
    class FakeApi:
        def __init__(self, token=None): pass
        def model_info(self, repo_id, token=None): return fake_info
    monkeypatch.setattr(loader_mod, "HfApi", FakeApi)

    loader = HFModelLoader(token=None)
    with pytest.raises(RuntimeError) as exc:
        loader.fetch_model_info("someone/diffuser")
    assert "Diffusers" in str(exc.value)


# -------------------------
# load_json tests
# -------------------------
def test_load_json_download_failure(monkeypatch):
    monkeypatch.setattr(loader_mod, "authenticate_hf", lambda token=None: "tok")
    monkeypatch.setattr(loader_mod, "hf_hub_download", lambda **kw: (_ for _ in ()).throw(Exception("nope")))
    loader = HFModelLoader(token=None)
    result = loader.load_json("some/repo", "config.json")
    assert result is None


def test_load_json_success(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_mod, "authenticate_hf", lambda token=None: "tok")
    # Create a temporary JSON file
    data = {"k": "v"}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    # Make hf_hub_download return that path
    monkeypatch.setattr(loader_mod, "hf_hub_download", lambda repo_id, filename, token=None: str(p))
    loader = HFModelLoader(token=None)
    loaded = loader.load_json("repo/x", "cfg.json")
    assert isinstance(loaded, dict)
    assert loaded["k"] == "v"


# -------------------------
# load_lora_info tests
# -------------------------
def test_load_lora_info_no_config(monkeypatch):
    # No adapter configs and no siblings => returns None
    monkeypatch.setattr(loader_mod, "authenticate_hf", lambda token=None: "tok")
    # load_json_quiet returns None
    monkeypatch.setattr(loader_mod.HFModelLoader, "load_json_quiet", lambda self, r, f: None)
    # fetch_model_info returns empty siblings
    monkeypatch.setattr(loader_mod.HFModelLoader, "fetch_model_info", lambda self, r: {"siblings": []})
    loader = HFModelLoader(token=None)
    assert loader.load_lora_info("no/lora") is None


def test_load_lora_info_with_config_and_siblings(monkeypatch):
    monkeypatch.setattr(loader_mod, "authenticate_hf", lambda token=None: "tok")

    # Simulate load_json_quiet returning a lora config
    monkeypatch.setattr(loader_mod.HFModelLoader, "load_json_quiet", lambda self, r, f: {"r": 4})

    # fetch_model_info returns siblings (filenames only)
    fake_fetch = lambda self, r: {"siblings": ["adapter_config.json", "a_weights.bin", "b.safetensors"]}
    monkeypatch.setattr(loader_mod.HFModelLoader, "fetch_model_info", fake_fetch)

    loader = HFModelLoader(token=None)
    info = loader.load_lora_info("some/lora")
    assert isinstance(info, dict)
    assert info.get("estimated_parameters") == 0
    for key in ("r", "alpha", "fan_in_fan_out", "target_modules"):
        assert key in info
