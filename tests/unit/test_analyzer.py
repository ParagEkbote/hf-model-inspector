import pytest
from hf_model_inspector.analyzer import (
    estimate_param_count,
    detect_quant_and_precision,
    analyze_tokenizer,
    extract_architecture_extras,
)


class TestEstimateParamCount:
    """Tests for estimate_param_count function."""

    def test_estimate_from_siblings_fp16(self):
        siblings = ["model-fp16.safetensors", "config.json"]
        count, method = estimate_param_count("repo/model", None, siblings)
        assert count is None
        assert "fp16" in method or "bf16" in method
        assert "shard_size_sum" in method

    def test_estimate_from_siblings_fp8(self):
        siblings = ["model-fp8.bin", "tokenizer.json"]
        count, method = estimate_param_count("repo/model", None, siblings)
        assert count is None
        assert "fp8" in method
        assert "shard_size_sum" in method

    def test_estimate_from_siblings_int_quant(self):
        siblings = ["model-gptq-int4.safetensors"]
        count, method = estimate_param_count("repo/model", None, siblings)
        assert count is None
        assert "int" in method

    def test_estimate_from_siblings_unknown_precision(self):
        siblings = ["model.bin", "config.json"]
        config = {"torch_dtype": "float16"}
        count, method = estimate_param_count("repo/model", config, siblings)
        assert count is None
        assert "fp16" in method or "unknown" in method

    def test_estimate_from_config_heuristic(self):
        config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "vocab_size": 30000,
        }
        count, method = estimate_param_count("repo/model", config, [])
        assert count is not None
        assert count > 0
        assert method == "config_heuristic"
        # Verify calculation: vocab*hidden + layers*(hidden*hidden*12)
        expected = 30000 * 768 + 12 * (768 * 768 * 12)
        assert count == expected

    def test_estimate_from_config_d_model(self):
        config = {
            "d_model": 512,
            "n_layer": 6,
            "n_vocab": 20000,
        }
        count, method = estimate_param_count("repo/model", config, [])
        assert count is not None
        assert count > 0
        assert method == "config_heuristic"

    def test_estimate_missing_required_fields(self):
        config = {"hidden_size": 768}  # missing num_hidden_layers
        count, method = estimate_param_count("repo/model", config, [])
        assert count is None
        assert method == "unknown"

    def test_estimate_no_config_no_siblings(self):
        count, method = estimate_param_count("repo/model", None, [])
        assert count is None
        assert method == "unknown"

    def test_estimate_config_exception_handling(self):
        config = {"hidden_size": "invalid", "num_hidden_layers": 12}
        count, method = estimate_param_count("repo/model", config, [])
        assert count is None
        assert method == "unknown"


class TestDetectQuantAndPrecision:
    """Tests for detect_quant_and_precision function."""

    def test_no_quantization(self):
        result = detect_quant_and_precision("repo/model", None, [], None)
        assert result["quantized"] is False
        assert result["quant_methods"] == []
        assert result["precision"] == "unknown"

    def test_quantization_config_json(self):
        def mock_load_json(repo_id, filename):
            if filename == "quantization_config.json":
                return {"method": "gptq", "bits": 4}
            return None

        result = detect_quant_and_precision(
            "repo/model", None, [], mock_load_json
        )
        assert result["quantized"] is True
        assert "gptq" in result["quant_methods"]

    def test_quantization_config_json_alternative_key(self):
        def mock_load_json(repo_id, filename):
            if filename == "quantization_config.json":
                return {"quantization_method": "awq"}
            return None

        result = detect_quant_and_precision(
            "repo/model", None, [], mock_load_json
        )
        assert result["quantized"] is True
        assert "awq" in result["quant_methods"]

    def test_detect_gptq_from_filename(self):
        siblings = ["model-gptq-4bit.safetensors"]
        result = detect_quant_and_precision("repo/model", None, siblings, None)
        assert result["quantized"] is True
        assert "gptq" in result["quant_methods"]

    def test_detect_bitsandbytes_from_filename(self):
        siblings = ["model-bnb-8bit.bin"]
        result = detect_quant_and_precision("repo/model", None, siblings, None)
        assert result["quantized"] is True
        assert "bitsandbytes" in result["quant_methods"]

    def test_detect_awq_from_filename(self):
        siblings = ["model-awq.safetensors"]
        result = detect_quant_and_precision("repo/model", None, siblings, None)
        assert result["quantized"] is True
        assert "awq" in result["quant_methods"]

    def test_detect_fp16_precision(self):
        siblings = ["model-fp16.safetensors"]
        result = detect_quant_and_precision("repo/model", None, siblings, None)
        assert result["precision"] == "fp16"

    def test_detect_bf16_precision(self):
        siblings = ["model-bf16.bin"]
        result = detect_quant_and_precision("repo/model", None, siblings, None)
        assert result["precision"] == "bf16"

    def test_detect_fp8_precision(self):
        siblings = ["model-fp8.safetensors"]
        result = detect_quant_and_precision("repo/model", None, siblings, None)
        assert result["precision"] == "fp8"

    def test_detect_int8_precision(self):
        siblings = ["model-int8.bin"]
        result = detect_quant_and_precision("repo/model", None, siblings, None)
        assert result["precision"] == "int8"

    def test_detect_int4_precision(self):
        siblings = ["model-int4.safetensors"]
        result = detect_quant_and_precision("repo/model", None, siblings, None)
        assert result["precision"] == "int4"

    def test_precision_from_config_torch_dtype(self):
        config = {"torch_dtype": "float16"}
        result = detect_quant_and_precision("repo/model", config, [], None)
        assert result["precision"] == "fp16"

    def test_precision_from_config_dtype(self):
        config = {"dtype": "bfloat16"}
        result = detect_quant_and_precision("repo/model", config, [], None)
        assert result["precision"] == "fp16"

    def test_precision_from_config_fp32(self):
        config = {"torch_dtype": "float32"}
        result = detect_quant_and_precision("repo/model", config, [], None)
        assert result["precision"] == "fp32"

    def test_multiple_quant_methods(self):
        def mock_load_json(repo_id, filename):
            if filename == "quantization_config.json":
                return {"method": "gptq"}
            return None

        siblings = ["model-awq.safetensors"]
        result = detect_quant_and_precision(
            "repo/model", None, siblings, mock_load_json
        )
        assert result["quantized"] is True
        assert len(result["quant_methods"]) == 2
        assert "gptq" in result["quant_methods"]
        assert "awq" in result["quant_methods"]


class TestAnalyzeTokenizer:
    """Tests for analyze_tokenizer function."""

    def test_no_tokenizer(self):
        result = analyze_tokenizer(None)
        assert result["present"] is False

    def test_empty_tokenizer(self):
        # Empty dict is truthy, so it should return present=True with None values
        result = analyze_tokenizer({})
        assert result["present"] is True
        assert result["type"] is None
        assert result["vocab_size"] is None
        assert result["model_max_length"] is None
        assert result["special_tokens"] == []

    def test_tokenizer_with_model_vocab_dict(self):
        tokenizer = {
            "model": {
                "type": "BPE",
                "vocab": {"hello": 0, "world": 1, "test": 2}
            }
        }
        result = analyze_tokenizer(tokenizer)
        assert result["present"] is True
        assert result["type"] == "BPE"
        assert result["vocab_size"] == 3

    def test_tokenizer_with_model_vocab_list(self):
        tokenizer = {
            "model": {
                "model_type": "WordPiece",
                "vocab": ["[PAD]", "[UNK]", "[CLS]"]
            }
        }
        result = analyze_tokenizer(tokenizer)
        assert result["present"] is True
        assert result["type"] == "WordPiece"
        assert result["vocab_size"] == 3

    def test_tokenizer_config_style(self):
        tokenizer = {
            "tokenizer_class": "BertTokenizer",
            "model_max_length": 512,
            "truncation": True,
            "do_lower_case": False,
        }
        result = analyze_tokenizer(tokenizer)
        assert result["present"] is True
        assert result["model_max_length"] == 512
        assert result["truncation"] is True
        assert result["lowercase"] is False

    def test_special_tokens_from_dict(self):
        tokenizer = {
            "added_tokens": {
                "[PAD]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
            }
        }
        result = analyze_tokenizer(tokenizer)
        assert result["present"] is True
        assert len(result["special_tokens"]) == 3
        assert "[PAD]" in result["special_tokens"]

    def test_special_tokens_from_list_with_dicts(self):
        tokenizer = {
            "added_tokens_decoder": [
                {"content": "[PAD]", "id": 0},
                {"token": "[CLS]", "id": 1},
            ]
        }
        result = analyze_tokenizer(tokenizer)
        assert result["present"] is True
        assert len(result["special_tokens"]) == 2
        assert "[PAD]" in result["special_tokens"]
        assert "[CLS]" in result["special_tokens"]

    def test_special_tokens_from_special_tokens_map(self):
        tokenizer = {
            "special_tokens_map": {
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
            }
        }
        result = analyze_tokenizer(tokenizer)
        assert result["present"] is True
        assert len(result["special_tokens"]) == 2

    def test_normalizer(self):
        tokenizer = {
            "normalizer": {
                "type": "BertNormalizer",
                "lowercase": True,
            }
        }
        result = analyze_tokenizer(tokenizer)
        assert result["present"] is True
        assert result["normalization"] is not None
        assert result["normalization"]["type"] == "BertNormalizer"

    def test_full_tokenizer_config(self):
        tokenizer = {
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1}
            },
            "tokenizer_class": "GPT2Tokenizer",
            "model_max_length": 1024,
            "truncation": True,
            "do_lower_case": True,
            "added_tokens": {"<|endoftext|>": 50256},
            "normalizer": {"type": "NFC"},
        }
        result = analyze_tokenizer(tokenizer)
        assert result["present"] is True
        assert result["type"] == "BPE"
        assert result["vocab_size"] == 2
        assert result["model_max_length"] == 1024
        assert result["truncation"] is True
        assert result["lowercase"] is True
        assert len(result["special_tokens"]) == 1
        assert result["normalization"] is not None


class TestExtractArchitectureExtras:
    """Tests for extract_architecture_extras function."""

    def test_no_config(self):
        result = extract_architecture_extras(None)
        assert result == {}

    def test_empty_config(self):
        result = extract_architecture_extras({})
        assert result == {}

    def test_extract_basic_fields(self):
        config = {
            "intermediate_size": 3072,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "layer_norm_eps": 1e-12,
        }
        result = extract_architecture_extras(config)
        assert result["intermediate_size"] == 3072
        assert result["hidden_dropout_prob"] == 0.1
        assert result["attention_probs_dropout_prob"] == 0.1
        assert result["layer_norm_eps"] == 1e-12

    def test_extract_activation_function(self):
        config = {"activation_function": "gelu"}
        result = extract_architecture_extras(config)
        assert result["activation_function"] == "gelu"

    def test_extract_rope_params(self):
        config = {
            "rope_theta": 10000,
            "rope_scaling": {"type": "linear", "factor": 2.0},
        }
        result = extract_architecture_extras(config)
        assert result["rope_theta"] == 10000
        assert result["rope_scaling"]["type"] == "linear"

    def test_extract_sliding_window(self):
        config = {"sliding_window": 4096}
        result = extract_architecture_extras(config)
        assert result["sliding_window"] == 4096

    def test_extract_cache_settings(self):
        config = {
            "use_cache": True,
            "tie_word_embeddings": False,
        }
        result = extract_architecture_extras(config)
        assert result["use_cache"] is True
        assert result["tie_word_embeddings"] is False

    def test_extract_kv_heads(self):
        config = {
            "num_key_value_heads": 8,
            "kv_head_dim": 64,
        }
        result = extract_architecture_extras(config)
        assert result["num_key_value_heads"] == 8
        assert result["kv_head_dim"] == 64

    def test_detect_rms_norm(self):
        config = {"rms_norm": True}
        result = extract_architecture_extras(config)
        assert result["norm_type"] == "RMSNorm"

    def test_detect_rms_norm_from_norm_type(self):
        config = {"norm_type": "rms_norm"}
        result = extract_architecture_extras(config)
        assert result["norm_type"] == "RMSNorm"

    def test_detect_layer_norm_from_norm_type(self):
        config = {"norm_type": "layer_norm"}
        result = extract_architecture_extras(config)
        assert result["norm_type"] == "LayerNorm"

    def test_detect_layer_norm_from_eps(self):
        config = {"layer_norm_eps": 1e-5}
        result = extract_architecture_extras(config)
        assert result["norm_type"] == "LayerNorm"
        assert result["layer_norm_eps"] == 1e-5

    def test_norm_type_not_inferred_without_hints(self):
        config = {"hidden_size": 768}
        result = extract_architecture_extras(config)
        assert "norm_type" not in result

    def test_extract_all_fields(self):
        config = {
            "intermediate_size": 3072,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "layer_norm_eps": 1e-12,
            "activation_function": "gelu",
            "rope_theta": 10000,
            "rope_scaling": {"type": "linear"},
            "sliding_window": 4096,
            "use_cache": True,
            "tie_word_embeddings": False,
            "num_key_value_heads": 8,
            "kv_head_dim": 64,
            "rms_norm": True,
        }
        result = extract_architecture_extras(config)
        assert len(result) == 13  # 12 keys + norm_type
        assert result["norm_type"] == "RMSNorm"
        assert result["intermediate_size"] == 3072
        assert result["rope_theta"] == 10000

    def test_partial_config(self):
        config = {
            "activation_function": "silu",
            "use_cache": False,
        }
        result = extract_architecture_extras(config)
        assert len(result) == 2
        assert result["activation_function"] == "silu"
        assert result["use_cache"] is False