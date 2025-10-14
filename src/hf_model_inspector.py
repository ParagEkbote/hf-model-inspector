from typing import Dict, Any, Optional, List
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
    loader = HFModelLoader(repo_id=repo_id, token=token)
    model, tokenizer = loader.load_model_and_tokenizer()

    # Core analysis
    param_info = estimate_param_count(model)
    quant_info = detect_quant_and_precision(model)
    tokenizer_info = analyze_tokenizer(tokenizer)
    arch_extras = extract_architecture_extras(model)

    return {
        "repo_id": repo_id,
        "architecture": model.__class__.__name__,
        "parameters": param_info,
        "quantization": quant_info,
        "tokenizer": tokenizer_info,
        "extras": arch_extras,
    }


def get_model_report_md(repo_id: str, token: Optional[str] = None) -> str:
    """Return model report formatted as Markdown string."""
    report_json = get_model_report_json(repo_id, token=token)
    return format_markdown(report_json)


def save_model_report(
    repo_id: str,
    md_path: Optional[str] = None,
    token: Optional[str] = None
) -> None:
    """Generate report and save as Markdown."""
    md_report = get_model_report_md(repo_id, token=token)
    save_outputs(md_report, md_path or f"{repo_id.replace('/', '_')}_report.md")


def get_lora_info(repo_id: str, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch LoRA-specific information for a given model."""
    loader = HFModelLoader(repo_id=repo_id, token=token)
    model, _ = loader.load_model_and_tokenizer()

    # Simple heuristic for LoRA adapters
    lora_modules = {
        name: str(param.dtype)
        for name, param in model.named_parameters()
        if "lora" in name.lower()
    }

    if not lora_modules:
        return None

    return {
        "num_lora_modules": len(lora_modules),
        "lora_module_names": list(lora_modules.keys()),
    }

