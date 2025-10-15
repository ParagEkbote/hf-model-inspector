import json
import logging
import subprocess
from typing import Any, Optional

from huggingface_hub import HfApi, HfFolder, hf_hub_download, whoami

logger = logging.getLogger(__name__)


def authenticate_hf(token: Optional[str] = None) -> str:
    """Authenticate with Hugging Face Hub securely and return token."""
    if token:
        return token.strip()

    cached_token = HfFolder.get_token()
    if cached_token:
        try:
            user_info = whoami(token=cached_token)
            logger.info(f"Authenticated as: {user_info.get('name', 'unknown')}")
            return cached_token
        except Exception:
            logger.warning("Cached token invalid or expired. Re-authenticating...")

    print("No valid Hugging Face login found. Launching `hf auth login`...")
    try:
        subprocess.run(["hf", "auth", "login"], check=True)
    except FileNotFoundError as err:
        raise RuntimeError(
            "Hugging Face CLI not found. Install via: pip install huggingface_hub"
        ) from err
    except subprocess.CalledProcessError as err:
        raise RuntimeError("Login process failed or cancelled by user.") from err

    new_token = HfFolder.get_token()
    if not new_token:
        raise RuntimeError(
            "Login unsuccessful. Please run `hf auth login` manually."
        ) from None

    logger.info("Successfully authenticated with Hugging Face (token kept private)")
    return new_token


class HFModelLoader:
    """Secure loader for Hugging Face Hub models (metadata & JSON)."""

    def __init__(self, token: Optional[str] = None):
        self.token = authenticate_hf(token)
        self.api = HfApi(token=self.token)

    def fetch_model_info(self, repo_id: str) -> Optional[dict[str, Any]]:
        try:
            info = self.api.model_info(repo_id, token=self.token)
        except Exception as err:
            logger.warning(f"Failed to fetch model info for repo: {repo_id}")
            raise RuntimeError(
                f"Unable to retrieve model info for '{repo_id}'"
            ) from err

        if getattr(info, "pipeline_tag", None) == "diffusers":
            raise RuntimeError(
                f"Diffusers model '{repo_id}' detected. Only metadata inspection is allowed."
            ) from None

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

    def load_json(self, repo_id: str, filename: str) -> Optional[dict[str, Any]]:
        """
        Try to load JSON from repo. If missing, return None instead of raising.
        """
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, token=self.token)
        except Exception:
            logger.debug(
                f"Could not download '{filename}' from '{repo_id}' (returning None)"
            )
            return None

        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.debug(
                f"Could not parse JSON '{filename}' in '{repo_id}' (returning None)"
            )
            return None

    def load_json_quiet(self, repo_id: str, filename: str) -> Optional[dict[str, Any]]:
        """Alias for load_json (quietly handled in calling code)."""
        return self.load_json(repo_id, filename)

    def load_lora_info(self, repo_id: str) -> Optional[dict[str, Any]]:
        """
        Load LoRA adapter metadata only (do NOT download weights). Returns None if configs missing.
        """
        lora_info = {}
        for cfg_name in ["adapter_config.json", "lora_config.json"]:
            cfg = self.load_json_quiet(repo_id, cfg_name)
            if cfg:
                lora_info.update(cfg)
                break

        if not lora_info:
            logger.debug(f"No LoRA config found for '{repo_id}'")
            return None

        model_info = self.fetch_model_info(repo_id)
        if not model_info:
            return None

        siblings = model_info.get("siblings", [])
        bin_files = [f for f in siblings if f.endswith(".bin")]
        logger.debug(f"LoRA bin files found (download skipped): {bin_files}")

        lora_info["estimated_parameters"] = 0
        lora_info["approx_precision_bytes"] = 4
        for key in ["r", "alpha", "fan_in_fan_out", "target_modules"]:
            if key not in lora_info:
                lora_info[key] = None

        return lora_info
        return lora_info
