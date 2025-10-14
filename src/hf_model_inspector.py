from typing import Dict, Any, Optional
from .loader import HFModelLoader
from .analyzer import (
    estimate_param_count,
    detect_quant_and_precision,
    analyze_tokenizer,
    extract_architecture_extras,
)
from .formatter import format_markdown, save_outputs

def get_model_report_json(repo_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """Return structured model report as a dict."""
    ...

def get_model_report_md(repo_id: str, token: Optional[str] = None) -> str:
    """Return model report formatted as Markdown string."""
    ...

def save_model_report(repo_id: str, md_path: Optional[str] = None, token: Optional[str] = None) -> None:
    """Generate report and save as Markdown."""
    ...

def get_lora_info(repo_id: str, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch LoRA-specific information for a given model."""
    ...

def recommend_models_for_gpu(gpu_specs: Dict[str, Any]) -> list[str]:
    """Return recommended models based on GPU memory, compute, etc."""
    ...