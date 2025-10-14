from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def estimate_param_count(
    repo_id: str, config: Optional[Dict], siblings: List[str]
) -> Tuple[Optional[int], str]:
    """
    Estimate parameter count for a model.
    Returns (param_count_estimate, method_description)

    Strategy:
      1. Try index.json parsing (preferred, if available)
      2. Sum shard file sizes and convert to parameter estimate using dtype heuristics
      3. Fallback to config-based heuristic
    """
    # Fallback since index.json parsing not implemented in this module
    # 2) sum shard bytes
    bytes_total = 0
    if siblings:
        joined = " ".join(siblings).lower()
        bytes_per_param = 2  # default
        if "fp16" in joined or "float16" in joined or "bf16" in joined:
            precision = "fp16/bf16"
            bytes_per_param = 2
        elif "fp8" in joined:
            precision = "fp8"
            bytes_per_param = 1
        elif "int8" in joined or "int4" in joined or "gptq" in joined:
            precision = "int"
            bytes_per_param = 1
        else:
            precision = "unknown"
            if config:
                cfg_dtype = config.get("torch_dtype") or config.get("dtype")
                if isinstance(cfg_dtype, str):
                    if "16" in cfg_dtype:
                        precision = "fp16"
                        bytes_per_param = 2
                    elif "8" in cfg_dtype:
                        precision = "fp8_or_int"
                        bytes_per_param = 1
                    elif "32" in cfg_dtype:
                        precision = "fp32"
                        bytes_per_param = 4
        # Estimation cannot actually sum file sizes here, so returning None
        return None, f"shard_size_sum ({precision})"

    # 3) fallback: config heuristics
    if config:
        try:
            h = config.get("hidden_size") or config.get("d_model") or 0
            l = config.get("num_hidden_layers") or config.get("n_layer") or 0
            v = config.get("vocab_size") or config.get("n_vocab") or 0
            if h and l:
                approx = v * h + l * (h * h * 12)
                return int(approx), "config_heuristic"
        except Exception:
            pass

    return None, "unknown"


def detect_quant_and_precision(
    repo_id: str, config: Optional[Dict], siblings: List[str], load_json_quiet=None
) -> Dict[str, Any]:
    """
    Detect quantization and precision.

    Returns:
      {
          "quantized": bool,
          "quant_methods": [...],
          "precision": "fp16|bf16|fp8|int8|unknown"
      }
    """
    result = {"quantized": False, "quant_methods": [], "precision": "unknown"}

    # check quantization config if loader function provided
    if load_json_quiet:
        qconf = load_json_quiet(repo_id, "quantization_config.json")
        if qconf:
            result["quantized"] = True
            m = qconf.get("method") or qconf.get("quantization_method")
            result["quant_methods"].append(m or "unknown")

    # filename hints
    joined = " ".join(siblings).lower() if siblings else ""
    for method, keywords in {
        "gptq": ["gptq"],
        "bitsandbytes": ["bnb", "bitsandbytes"],
        "awq": ["awq"],
    }.items():
        if any(k in joined for k in keywords):
            result["quantized"] = True
            result["quant_methods"].append(method)

    # detect precision by filename
    for p, keywords in {
        "fp16": ["fp16", "float16"],
        "bf16": ["bf16"],
        "fp8": ["fp8"],
        "int8": ["int8"],
        "int4": ["int4"],
    }.items():
        if any(k in joined for k in keywords):
            result["precision"] = p
            break

    # try config hints
    if result["precision"] == "unknown" and config:
        cfg_dtype = config.get("torch_dtype") or config.get("dtype") or config.get("torch_dtype_str")
        if isinstance(cfg_dtype, str):
            if "16" in cfg_dtype:
                result["precision"] = "fp16"
            elif "8" in cfg_dtype:
                result["precision"] = "fp8"
            elif "32" in cfg_dtype:
                result["precision"] = "fp32"

    return result


def analyze_tokenizer(tokenizer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze tokenizer config."""
    if not tokenizer:
        return {"present": False}

    info = {
        "present": True,
        "type": None,
        "vocab_size": None,
        "model_max_length": None,
        "special_tokens": [],
        "truncation": None,
        "normalization": None,
        "lowercase": None,
    }

    model_part = tokenizer.get("model") if isinstance(tokenizer, dict) else None
    if model_part:
        info["type"] = model_part.get("type") or model_part.get("model_type")
        vocab = model_part.get("vocab")
        if isinstance(vocab, dict):
            info["vocab_size"] = len(vocab)
        elif isinstance(vocab, list):
            info["vocab_size"] = len(vocab)

    # tokenizer_config.json style keys
    for k in ["tokenizer_class", "model_max_length", "truncation", "do_lower_case"]:
        if k in tokenizer:
            info_key = "lowercase" if k == "do_lower_case" else k
            info[info_key] = tokenizer[k]

    # special tokens
    at = tokenizer.get("added_tokens") or tokenizer.get("added_tokens_decoder") or tokenizer.get("special_tokens_map")
    if isinstance(at, dict):
        info["special_tokens"] = list(at.keys())
    elif isinstance(at, list):
        toks = []
        for t in at:
            if isinstance(t, dict):
                toks.append(t.get("content") or t.get("token"))
        info["special_tokens"] = toks

    # normalizer
    if "normalizer" in tokenizer:
        info["normalization"] = tokenizer.get("normalizer")

    return info


def extract_architecture_extras(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract additional model config information."""
    extras = {}
    if not config:
        return extras

    keys = [
        "intermediate_size",
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
        "layer_norm_eps",
        "activation_function",
        "rope_theta",
        "rope_scaling",
        "sliding_window",
        "use_cache",
        "tie_word_embeddings",
        "num_key_value_heads",
        "kv_head_dim",
    ]

    for k in keys:
        if k in config:
            extras[k] = config[k]

    # layer norm type inference
    norm_type = None
    if config.get("rms_norm") or "rms" in str(config.get("norm_type", "")).lower():
        norm_type = "RMSNorm"
    elif "layer_norm" in str(config.get("norm_type", "")).lower() or "layernorm" in str(config.get("norm_type", "")).lower():
        norm_type = "LayerNorm"
    elif "layer_norm_eps" in config:
        norm_type = "LayerNorm"

    if norm_type:
        extras["norm_type"] = norm_type

    return extras
