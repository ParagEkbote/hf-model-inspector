#Todo: 1. Lora should not be downloaded, only configs should be printed
# 2. fix token leakage issue and give error for diffusers.
from typing import Dict, Any, Optional
from huggingface_hub import HfApi, hf_hub_download, HfFolder, whoami
import json
import logging
import subprocess
import os

logger = logging.getLogger(__name__)


def authenticate_hf(token: Optional[str] = None) -> str:
    """
    Authenticate with Hugging Face Hub securely.

    Priority:
    1. Use the provided token
    2. Use cached token from `hf auth login`
    3. Prompt interactive login if missing or invalid

    Returns:
        A valid Hugging Face token (kept private)
    """
    # 1. Direct token provided
    if token:
        return token.strip()

    # 2. Check cached token
    cached_token = HfFolder.get_token()
    if cached_token:
        try:
            user_info = whoami(token=cached_token)
            logger.info(f"Authenticated as: {user_info.get('name', 'unknown')}")
            return cached_token
        except Exception:
            logger.warning("Cached token invalid or expired. Re-authenticating...")

    # 3. Launch CLI login if token missing/invalid
    print("No valid Hugging Face login found. Launching `hf auth login`...")
    try:
        subprocess.run(["hf", "auth", "login"], check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "Hugging Face CLI not found. Install via: pip install huggingface_hub"
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("Login process failed or cancelled by user.")

    # 4. Try fetching the new token
    new_token = HfFolder.get_token()
    if not new_token:
        raise RuntimeError("Login unsuccessful. Please run `hf auth login` manually.")
    
    logger.info("Successfully authenticated with Hugging Face (token kept private)")
    return new_token


class HFModelLoader:
    """Secure loader for Hugging Face Hub models (metadata & JSON)."""

    def __init__(self, token: Optional[str] = None):
        self.token = authenticate_hf(token)
        self.api = HfApi(token=self.token)

    def fetch_model_info(self, repo_id: str) -> Optional[Dict[str, Any]]:
        try:
            info = self.api.model_info(repo_id, token=self.token)
        except Exception:
            logger.warning(f"Failed to fetch model info for repo: {repo_id}")
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
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, token=self.token)
        except Exception:
            logger.debug(f"Could not download file '{filename}' from repo '{repo_id}'")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.debug(f"Could not parse JSON file '{filename}'")
            return None

    def load_json_quiet(self, repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
        return self.load_json(repo_id, filename)

    def load_lora_info(self, repo_id: str) -> Optional[Dict[str, Any]]:
        lora_info = {}

        # Load adapter config if exists
        for cfg_name in ["adapter_config.json", "lora_config.json"]:
            cfg = self.load_json_quiet(repo_id, cfg_name)
            if cfg:
                lora_info.update(cfg)
                break

        # Collect .bin files for parameter estimation
        model_info = self.fetch_model_info(repo_id)
        if not model_info:
            return None

        siblings = model_info.get("siblings", [])
        bin_files = [f for f in siblings if f.endswith(".bin")]

        if not lora_info and not bin_files:
            return None

        total_bytes = 0
        for bin_file in bin_files:
            try:
                path = hf_hub_download(repo_id=repo_id, filename=bin_file, token=self.token)
                total_bytes += os.path.getsize(path)
            except Exception:
                logger.debug(f"Could not get size for file '{bin_file}'")
                continue

        lora_info["estimated_parameters"] = total_bytes // 4
        lora_info["approx_precision_bytes"] = 4

        # Add default keys
        for key in ["r", "alpha", "fan_in_fan_out", "target_modules"]:
            if key not in lora_info:
                lora_info[key] = None

        return lora_info
