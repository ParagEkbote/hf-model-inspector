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

def recommend_models_for_gpu(gpu_specs: Dict[str, any]) -> List[str]:
    """
    Recommend model sizes/types based on GPU specs.
    
    gpu_specs: {
        "name": str,                # e.g., "A100", "RTX3090"
        "memory_gb": int,           # VRAM in GB
        "compute_capability": float # CUDA compute capability
    }
    
    Returns a list of recommended model "size" categories.
    """
    memory_gb = gpu_specs.get("memory_gb", 8)
    compute_cap = gpu_specs.get("compute_capability", 7.0)
    gpu_name = gpu_specs.get("name", "").lower()
    
    recommendations = []

    # Low memory GPUs (8–12GB)
    if memory_gb < 12:
        recommendations.append("small")   # e.g., distilled or tiny models
    # Mid-range GPUs (12–24GB)
    if 12 <= memory_gb < 24:
        recommendations.append("medium")  # e.g., base LLMs, 3–7B models
    # High-end GPUs (24GB+)
    if memory_gb >= 24:
        recommendations.append("large")   # e.g., 13B+ models or multi-GPU setups

    # Adjust recommendations by GPU type
    if "a100" in gpu_name or "h100" in gpu_name:
        # Tensor core GPUs: can handle larger models efficiently
        if "medium" not in recommendations:
            recommendations.append("medium")
        if "large" not in recommendations:
            recommendations.append("large")
    elif "rtx" in gpu_name or "v100" in gpu_name:
        # Consumer/older GPUs: mostly small or medium
        if "large" in recommendations:
            recommendations.remove("large")

    # Always include a fallback
    if not recommendations:
        recommendations.append("small")

    return recommendations