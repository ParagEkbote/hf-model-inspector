from typing import Any, Optional

from .analyzer import (
    analyze_tokenizer,
    detect_quant_and_precision,
    estimate_param_count,
    extract_architecture_extras,
)
from .formatter import format_markdown, save_outputs
from .loader import HFModelLoader


def get_model_report_json(repo_id: str, token: Optional[str] = None) -> dict[str, Any]:
    """Return structured model report as a dict (without downloading weights)."""
    loader = HFModelLoader(token=token)

    # Fetch metadata and config files only (no model weights!)
    model_info = loader.fetch_model_info(repo_id)
    config = loader.load_json(repo_id, "config.json")
    tokenizer_config = loader.load_json(repo_id, "tokenizer_config.json")

    if not config:
        raise ValueError(f"Could not load config.json for {repo_id}")

    # Get siblings (list of files in the repo)
    siblings = model_info.get("siblings", []) if model_info else []

    # Analyze using the correct function signatures
    param_count, param_method = estimate_param_count(repo_id, config, siblings)
    quant_info = detect_quant_and_precision(
        repo_id, config, siblings, load_json_quiet=loader.load_json_quiet
    )
    tokenizer_info = analyze_tokenizer(tokenizer_config)
    arch_extras = extract_architecture_extras(config)

    # Format parameter info
    param_info = {
        "total": param_count,
        "total_millions": round(param_count / 1_000_000, 2) if param_count else None,
        "total_billions": (
            round(param_count / 1_000_000_000, 3) if param_count else None
        ),
        "estimation_method": param_method,
    }

    return {
        "repo_id": repo_id,
        "architecture": (
            config.get("architectures", ["Unknown"])[0]
            if config.get("architectures")
            else "Unknown"
        ),
        "model_type": config.get("model_type", "Unknown"),
        "parameters": param_info,
        "quantization": quant_info,
        "tokenizer": tokenizer_info,
        "architecture_extras": arch_extras,
        "metadata": {
            "downloads": model_info.get("downloads", 0) if model_info else 0,
            "likes": model_info.get("likes", 0) if model_info else 0,
            "tags": model_info.get("tags", []) if model_info else [],
            "library": model_info.get("library_name") if model_info else None,
            "pipeline_tag": model_info.get("pipeline_tag") if model_info else None,
        },
    }


def get_model_report_md(repo_id: str, token: Optional[str] = None) -> str:
    """Return model report formatted as Markdown string."""
    report_json = get_model_report_json(repo_id, token=token)
    return format_markdown(report_json)


def save_model_report(
    repo_id: str, md_path: Optional[str] = None, token: Optional[str] = None
) -> None:
    """Generate report and save as Markdown."""
    md_report = get_model_report_md(repo_id, token=token)
    save_outputs(md_report, md_path or f"{repo_id.replace('/', '_')}_report.md")


def get_lora_info(repo_id: str, token: Optional[str] = None) -> Optional[dict[str, Any]]:
    """
    Fetch detailed LoRA configuration for a given model repository.

    Returns:
        A dictionary containing LoRA adapter parameters (e.g., lora_alpha, r, target_modules)
        or None if no LoRA configuration is found.
    """
    loader = HFModelLoader(token=token)
    lora_cfg = loader.load_lora_info(repo_id)

    if not lora_cfg:
        return None

    # Normalize & summarize key LoRA parameters
    summary = {
        "repo_id": repo_id,
        "peft_type": lora_cfg.get("peft_type", "LORA"),
        "task_type": lora_cfg.get("task_type", "Unknown"),
        "base_model_name_or_path": lora_cfg.get("base_model_name_or_path"),
        "r": lora_cfg.get("r"),
        "lora_alpha": lora_cfg.get("lora_alpha"),
        "lora_dropout": lora_cfg.get("lora_dropout", 0),
        "bias": lora_cfg.get("bias"),
        "fan_in_fan_out": lora_cfg.get("fan_in_fan_out", False),
        "use_dora": lora_cfg.get("use_dora", False),
        "use_qalora": lora_cfg.get("use_qalora", False),
        "use_rslora": lora_cfg.get("use_rslora", False),
        "target_modules": lora_cfg.get("target_modules", []),
        "auto_mapping": lora_cfg.get("auto_mapping", {}),
        "rank_pattern": lora_cfg.get("rank_pattern", {}),
        "alpha_pattern": lora_cfg.get("alpha_pattern", {}),
        "megatron_core": lora_cfg.get("megatron_core", None),
        "qalora_group_size": lora_cfg.get("qalora_group_size", None),
        "approx_precision_bytes": lora_cfg.get("approx_precision_bytes", 4),
        "estimated_parameters": lora_cfg.get("estimated_parameters", 0),
    }

    return summary


def recommend_models_for_gpu(gpu_specs: dict[str, Any]) -> list[str]:
    """
    Recommend model sizes/types based on GPU specs.

    Args:
        gpu_specs: Dictionary with optional keys:
            - name (str): GPU model name, e.g. "A100", "RTX 3090"
            - memory_gb (int): GPU VRAM size
            - compute_capability (float): CUDA compute capability (e.g., 6.1, 8.0)

    Returns:
        list[str]: Recommended model size categories (small, medium, large)
    """
    memory_gb = gpu_specs.get("memory_gb", 8)
    compute_cap = gpu_specs.get("compute_capability", 7.0)
    gpu_name = gpu_specs.get("name", "").lower()

    recommendations: list[str] = []

    # 🧠 Base recommendation by memory size
    if memory_gb < 12:
        recommendations.append("small")
    elif 12 <= memory_gb < 24:
        recommendations.append("medium")
    else:
        recommendations.append("large")

    # ⚙️ Adjust based on compute capability
    if compute_cap < 7.0:
        # Pre-Volta GPUs (Pascal and older)
        recommendations = ["small"]
    elif 7.0 <= compute_cap < 8.0:
        # Volta / Turing
        if "large" in recommendations:
            recommendations.remove("large")
    elif compute_cap >= 8.0:
        # Ampere / Hopper architectures
        if "medium" not in recommendations:
            recommendations.append("medium")
        if "large" not in recommendations:
            recommendations.append("large")

    # 💡 Fine-tune based on GPU name
    if "a100" in gpu_name or "h100" in gpu_name:
        # Enterprise GPUs
        recommendations = ["medium", "large"]
    elif "rtx" in gpu_name and memory_gb < 24:
        # Consumer RTX cards (e.g., 3060, 3080)
        if "large" in recommendations:
            recommendations.remove("large")

    # 🛡️ Always include fallback
    if not recommendations:
        recommendations.append("small")

    return sorted(set(recommendations))
