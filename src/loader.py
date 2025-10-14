from typing import Dict, Any, Optional, List
from huggingface_hub import (
    HfApi,
    hf_hub_download,
    HfFolder,
    RepositoryNotFoundError,
    whoami,
)
import json
import logging
import subprocess

logger = logging.getLogger(__name__)


def authenticate_hf(token: Optional[str] = None) -> str:
    """
    Authenticate with Hugging Face Hub.

    Priority:
    1. Use the provided token
    2. Use cached token from `hf auth login`
    3. Prompt interactive login if missing or invalid

    Returns:
        str: Valid authentication token.
    """
    # 1. Direct token provided
    if token:
        return token.strip()

    # 2. Check cached token
    cached_token = HfFolder.get_token()
    if cached_token:
        try:
            _ = whoami(token=cached_token)
            return cached_token
        except Exception:
            logger.warning("Cached token invalid or expired. Re-authenticating...")

    # 3. Launch CLI login if token missing/invalid
    print("🔐 No valid Hugging Face login found. Launching `hf auth login`...")
    try:
        subprocess.run(["hf", "auth", "login"], check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "❌ Hugging Face CLI not found. Install via: pip install huggingface_hub"
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("❌ Login process failed or cancelled by user.")

    # 4. Try fetching the new token
    new_token = HfFolder.get_token()
    if not new_token:
        raise RuntimeError("❌ Login unsuccessful. Please run `hf auth login` manually.")
    return new_token


class HFModelLoader:
    """Handles downloading and retrieving model files and metadata from Hugging Face Hub."""

    def __init__(self, token: Optional[str] = None):
        """Initialize loader with optional Hugging Face token."""
        self.token = authenticate_hf(token)
        self.api = HfApi(token=self.token)

    def fetch_model_info(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """Fetch model metadata from the Hugging Face Hub."""
        try:
            info = self.api.model_info(repo_id, token=self.token)
        except RepositoryNotFoundError:
            logger.error(f"Repository not found: {repo_id}")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch model info for {repo_id}: {e}")
            return None

        return {
            "id": getattr(info, "modelId", repo_id),
            "sha": getattr(info, "sha", None),
            "downloads": getattr(info, "downloads", 0),
            "likes": getattr(info, "likes", 0),
            "tags": getattr(info, "tags", []),
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "library_name": getattr(info, "library_name", None),
            "private": getattr(info, "private", False),
            "gated": getattr(info, "gated", False),
            "author": getattr(info, "author", None),
            "siblings": [s.rfilename for s in getattr(info, "siblings", []) or []],
            "cardData": getattr(info, "cardData", None),
            "lastModified": getattr(info, "lastModified", None),
            "createdAt": getattr(info, "createdAt", None),
        }

    def load_json(self, repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Download and parse a JSON file from a repository."""
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, token=self.token)
        except Exception as e:
            logger.debug(f"Could not download {filename} from {repo_id}: {e}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.debug(f"Invalid JSON format in {filename} from {repo_id}")
        except FileNotFoundError:
            logger.debug(f"File not found after download: {filename}")
        return None

    def load_json_quiet(self, repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Load JSON quietly (no log spam on failure)."""
        return self.load_json(repo_id, filename)

    def load_lora_info(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """Load LoRA adapter metadata if available."""
        for fname in ["adapter_config.json", "lora_config.json", "adapter_model.bin"]:
            result = self.load_json_quiet(repo_id, fname)
            if result:
                return result
        return None
