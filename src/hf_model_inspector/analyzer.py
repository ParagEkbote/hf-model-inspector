import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

def estimate_param_count(
    _repo_id: str, config: Optional[dict], siblings: list[str]
) -> tuple[int, str]:
    """
    Estimate parameter count for a model.
    Returns (param_count_estimate, method_description)
    """
    # Try to estimate from config values first
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

    # Fallback: check siblings for precision hints
    if siblings:
        joined = " ".join(siblings).lower()
        if "fp16" in joined or "float16" in joined or "bf16" in joined:
            precision = "fp16/bf16"
        elif "fp8" in joined:
            precision = "fp8"
        elif "int8" in joined or "int4" in joined or "gptq" in joined:
            precision = "int"
        else:
            precision = "unknown"

        return 0, f"shard_size_sum ({precision})"

    return 0, "unknown"


def detect_quant_and_precision(
    repo_id: str, config: Optional[dict], siblings: list[str], load_json_quiet=None
) -> dict[str, Any]:
    """
    Detect quantization and precision from HuggingFace Hub models.

    Returns:
      {
          "quantized": bool,
          "quant_methods": [...],
          "precision": "fp16|bf16|fp8|int8|int4|fp32|unknown"
      }
    """
    result = {"quantized": False, "quant_methods": [], "precision": "unknown"}
    methods_found = set()

    # 1. Check quantization config
    if load_json_quiet:
        qconf = None
        for fname in ["quantization_config.json", "config.json"]:
            qconf = load_json_quiet(repo_id, fname)
            if qconf:
                # Look for quantization_config nested object
                if "quantization_config" in qconf:
                    qconf = qconf["quantization_config"]
                
                # Check for quantization indicators
                if qconf.get("quant_method") or qconf.get("quantization_method") or qconf.get("method"):
                    result["quantized"] = True
                    method = (qconf.get("quant_method") or 
                             qconf.get("quantization_method") or 
                             qconf.get("method") or "unknown")
                    methods_found.add(method.lower())
                
                # Check for bits field (common in quantized models)
                if qconf.get("bits"):
                    result["quantized"] = True
                    bits = qconf.get("bits")
                    if bits == 4:
                        methods_found.add("4bit")
                    elif bits == 8:
                        methods_found.add("8bit")
                
                # Check for load_in_4bit or load_in_8bit
                if qconf.get("load_in_4bit") or qconf.get("load_in_8bit"):
                    result["quantized"] = True
                    methods_found.add("bitsandbytes")
                
                # Check for torchao and quanto specific fields
                if qconf.get("quantization_scheme") or qconf.get("activations") or qconf.get("weights"):
                    result["quantized"] = True
                    # torchao often has 'quantization_scheme' field
                    if qconf.get("quantization_scheme"):
                        methods_found.add("torchao")
                
                # Quanto detection
                if qconf.get("quanto") or (qconf.get("method") and "quanto" in str(qconf.get("method")).lower()):
                    result["quantized"] = True
                    methods_found.add("quanto")
                
                # Get precision from config
                dtype = qconf.get("dtype") or qconf.get("torch_dtype")
                if isinstance(dtype, str):
                    result["precision"] = _parse_dtype(dtype)
                
                break

    # 2. Check main config for quantization hints
    if config:
        # Check for quantization_config in main config
        if "quantization_config" in config:
            result["quantized"] = True
            qc = config["quantization_config"]
            method = (qc.get("quant_method") or 
                     qc.get("quantization_method") or 
                     qc.get("method"))
            if method:
                methods_found.add(method.lower())
        
        # Check for other quantization indicators
        if config.get("quantization_method"):
            result["quantized"] = True
            methods_found.add(config["quantization_method"].lower())
        
        # Check for bits field
        if config.get("bits") or config.get("quantization_bits"):
            result["quantized"] = True
            bits = config.get("bits") or config.get("quantization_bits")
            if bits == 4:
                methods_found.add("4bit")
            elif bits == 8:
                methods_found.add("8bit")

    # 3. Filename-based detection (more comprehensive)
    joined = " ".join(siblings).lower() if siblings else ""
    
    # Enhanced quantization method detection
    method_patterns = {
        "gptq": ["gptq", "gptq-int4", "gptq-int8"],
        "awq": ["awq", "awq-int4"],
        "bitsandbytes": ["bnb", "bitsandbytes", "8bit", "4bit"],
        "gguf": ["gguf", ".gguf"],
        "ggml": ["ggml", ".ggml"],
        "exl2": ["exl2", "exllamav2"],
        "squeezellm": ["squeezellm"],
        "eetq": ["eetq"],
        "hqq": ["hqq", "half-quadratic"],
        "marlin": ["marlin"],
        "torchao": ["torchao", "torch-ao", "ao_quant"],
        "quanto": ["quanto"],
    }
    
    for method, keywords in method_patterns.items():
        if any(k in joined for k in keywords):
            result["quantized"] = True
            methods_found.add(method)

    # 4. Detect precision by filename
    if result["precision"] == "unknown":
        precision_patterns = {
            "fp16": ["fp16", "float16", "f16"],
            "bf16": ["bf16", "bfloat16"],
            "fp8": ["fp8", "float8"],
            "int8": ["int8", "8bit", "w8"],
            "int4": ["int4", "4bit", "w4"],
            "fp32": ["fp32", "float32", "f32"],
        }
        
        for prec, keywords in precision_patterns.items():
            if any(k in joined for k in keywords):
                result["precision"] = prec
                break

    # 5. Fallback: check config for precision
    if result["precision"] == "unknown" and config:
        cfg_dtype = (config.get("torch_dtype") or 
                    config.get("dtype") or 
                    config.get("torch_dtype_str"))
        if isinstance(cfg_dtype, str):
            result["precision"] = _parse_dtype(cfg_dtype)

    # 6. Clean up and deduplicate methods
    result["quant_methods"] = sorted(list(methods_found)) if methods_found else []
    
    # If no methods found but quantized flag is set, add "unknown"
    if result["quantized"] and not result["quant_methods"]:
        result["quant_methods"] = ["unknown"]

    return result


def _parse_dtype(dtype_str: str) -> str:
    """Parse dtype string to precision label."""
    dtype_str = dtype_str.lower()
    
    if "bfloat16" in dtype_str or "bf16" in dtype_str:
        return "bf16"
    elif "float16" in dtype_str or "fp16" in dtype_str:
        return "fp16"
    elif "float8" in dtype_str or "fp8" in dtype_str:
        return "fp8"
    elif "int8" in dtype_str:
        return "int8"
    elif "int4" in dtype_str:
        return "int4"
    elif "float32" in dtype_str or "fp32" in dtype_str:
        return "fp32"
    
    return "unknown"

def analyze_tokenizer(tokenizer: Optional[dict[str, Any]]) -> dict[str, Any]:
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
    at = (
        tokenizer.get("added_tokens")
        or tokenizer.get("added_tokens_decoder")
        or tokenizer.get("special_tokens_map")
    )
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


def extract_architecture_extras(config: Optional[dict[str, Any]]) -> dict[str, Any]:
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
    elif (
        "layer_norm" in str(config.get("norm_type", "")).lower()
        or "layernorm" in str(config.get("norm_type", "")).lower()
    ):
        norm_type = "LayerNorm"
    elif "layer_norm_eps" in config:
        norm_type = "LayerNorm"

    if norm_type:
        extras["norm_type"] = norm_type

    return extras
