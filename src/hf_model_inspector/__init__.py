# Core loader
# Analyzer functions
from .analyzer import (
    analyze_tokenizer,
    detect_quant_and_precision,
    estimate_param_count,
    extract_architecture_extras,
)

# Formatter functions
from .formatter import format_markdown, save_outputs

# Main API functions
from .hf_model_inspector import (
    get_lora_info,
    get_model_report_json,
    get_model_report_md,
    recommend_models_for_gpu,
    save_model_report,
)
from .loader import HFModelLoader
from .utils import field, humanize_params, safe_get

# Public API
__all__ = [
    "get_model_report_json",
    "get_model_report_md",  # Main API
    "save_model_report",
    "get_lora_info",
    "recommend_models_for_gpu",
    "HFModelLoader",  # Core classes
    "estimate_param_count",
    "detect_quant_and_precision",  # Analyzer
    "analyze_tokenizer",
    "extract_architecture_extras",
    "format_markdown",  # Formatter
    "save_outputs",
    "field",
    "safe_get",
    "humanize_params",
]
