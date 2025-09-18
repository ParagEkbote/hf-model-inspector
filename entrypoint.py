"""
HF Model Inspector — entrypoint

Features:
- dotenv support for local dev
- optional HF auth (for gated/private repos)
- handles custom architectures
- improved param counting using index.json or file sizes (no weight downloads)
- precise detection of precision (fp16/fp8/bf16) vs quantization (quantization_config.json)
- tokenizer details: type, special tokens, truncation, lowercasing/normalization
- extra architecture details: dropout, norm, activation, rope/alibi, cache, tie embeddings
- warnings for missing pieces and unsafe weight formats
- outputs Markdown report + JSON artifact
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set, Tuple

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, repo_exists, hf_hub_url
from huggingface_hub.errors import RepositoryNotFoundError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("hf-model-inspector")


# -------------------------
# Constants & helpers
# -------------------------
SUPPORTED_TRANSFORMERS = {
    # common model_type / architectures recognized by HF transformers
    "bert", "roberta", "gpt2", "gpt_neox", "llama", "mistral", "qwen", "gptj", "t5", "bart", "distilbert",
    "electra", "gpt_oss", "opt", "gpt", "mt5", "deberta", "camembert", "xlm", "unifiedqa", "clip"
}


def humanize_params(n: Optional[int]) -> str:
    if n is None:
        return "N/A"
    if n >= 1_000_000_000_000:
        return f"~{n/1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"~{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"~{n/1_000_000:.2f}M"
    return str(n)


def safe_get(d: dict, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        if k in d:
            return d[k]
    return default


# -------------------------
# Inspector class
# -------------------------
class ModelInspector:
    def __init__(self, token: Optional[str] = None):
        self.api = HfApi()
        self.token = token
        if token:
            # do not attempt to set git credential helper
            try:
                from huggingface_hub import login
                login(token=token, add_to_git_credential=False)
                logger.info("Authenticated to Hugging Face (token provided)")
            except Exception as e:
                logger.warning(f"Auth attempt failed: {e}")

    # ---- Hub metadata ----
    def fetch_model_info(self, repo_id: str) -> Optional[Dict[str, Any]]:
        try:
            info = self.api.model_info(repo_id, token=self.token)
            # extract the most useful fields
            result = {
                "id": getattr(info, "modelId", repo_id) if hasattr(info, "modelId") else repo_id,
                "sha": getattr(info, "sha", None),
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
                "tags": getattr(info, "tags", []),
                "pipeline_tag": getattr(info, "pipeline_tag", None),
                "library_name": getattr(info, "library_name", None),
                "private": getattr(info, "private", False),
                "gated": getattr(info, "gated", False),
                "author": getattr(info, "author", None),
                "siblings": [s.rfilename for s in getattr(info, "siblings", [])] if getattr(info, "siblings", None) else [],
                "cardData": getattr(info, "cardData", None),
                "lastModified": getattr(info, "lastModified", None),
                "createdAt": getattr(info, "createdAt", None),
            }
            return result
        except RepositoryNotFoundError:
            logger.error(f"Repository not found: {repo_id}")
            return None
        except Exception as e:
            logger.warning(f"Could not fetch model_info for {repo_id}: {e}")
            return None

    # ---- Load config/tokenizer ----
    def load_json(self, repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, token=self.token)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Could not download/parse {filename} from {repo_id}: {e}")
            return None

    # ---- Parameter counting ----
    def _parse_index_param_metadata(self, repo_id: str) -> Optional[int]:
        """
        Try to parse index json files usually present for sharded large models:
        - model.safetensors.index.json
        - pytorch_model.bin.index.json
        These may contain metadata.total_size in bytes (or param count in some repos).
        If total_size is bytes, we return it as approximate parameter bytes — attempt to convert:
          - if total_size looks like number of elements, return directly
          - else return None if ambiguous
        """
        index_candidates = [
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "pytorch_model-00001-of-*.bin.index.json",
            "model.safetensors.index",
        ]
        for idx in ["model.safetensors.index.json", "pytorch_model.bin.index.json", "model.safetensors.index.json"]:
            try:
                idx_path = hf_hub_download(repo_id=repo_id, filename=idx, token=self.token)
            except Exception:
                continue
            try:
                with open(idx_path, "r", encoding="utf-8") as f:
                    idx_json = json.load(f)
                # Many index files include "metadata" -> "total_size" as bytes
                metadata = idx_json.get("metadata", {}) if isinstance(idx_json, dict) else {}
                # Some index jsons include 'total_size' or 'total_tensors' or 'total_parameters'
                total_size = metadata.get("total_size") or metadata.get("total_parameters") or metadata.get("total_tensors")
                if isinstance(total_size, int):
                    # if it's bytes, we can't know params exactly; try heuristics:
                    # If index contains 'weight_map' with shapes, sum elements if available
                    # Some index.json include 'weights' mapping to shapes; attempt to sum num elements
                    total_params = None
                    weight_map = idx_json.get("weight_map") or idx_json.get("weights") or idx_json.get("mappings")
                    if isinstance(weight_map, dict):
                        # if shapes provided
                        elt_count = 0
                        for k, v in weight_map.items():
                            # v might be a filename or a dict with 'shape' or 'dtype'
                            if isinstance(v, dict) and "shape" in v:
                                shp = v.get("shape") or []
                                try:
                                    prod = 1
                                    for s in shp:
                                        prod *= int(s)
                                    elt_count += prod
                                except Exception:
                                    elt_count = None
                                    break
                        if elt_count and elt_count > 0:
                            return int(elt_count)
                    # fallback: if total_size huge (bytes) but no shapes, return None and let downstream sum shard sizes
                    return None
            except Exception as e:
                logger.debug(f"Failed to parse index json {idx_path}: {e}")
                continue
        return None

    def _sum_shard_file_sizes(self, repo_id: str, siblings: List[str]) -> Optional[int]:
        """
        Sum sizes of weight files (safetensors / .bin / .pt / .pth) using repo_file_info (no download).
        Returns total bytes across shards or None if none found.
        """
        total_bytes = 0
        found = False
        for fname in siblings:
            low = fname.lower()
            if low.endswith((".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx")) or "model-" in low:
                found = True
                try:
                    info = self.api.repo_file_info(repo_id, fname, token=self.token)
                    total_bytes += int(getattr(info, "size", 0))
                except Exception as e:
                    logger.debug(f"Could not stat {fname}: {e}")
                    # continue summing what we can
        if not found:
            return None
        return total_bytes

    def estimate_param_count(self, repo_id: str, config: Optional[Dict], siblings: List[str]) -> Tuple[Optional[int], str]:
        """
        Return (param_count_estimate, method_description)
        Strategy:
          1. Try index.json parsing for explicit parameter elements (preferred)
          2. Sum shard file sizes (bytes) and convert to parameter estimate using common dtype heuristics (fp16 ~2 bytes/param, bf16 ~2, fp32 ~4)
          3. Fallback to config-based heuristic (not reliable for custom architectures)
        """
        # 1) index-based
        idx_params = self._parse_index_param_metadata(repo_id)
        if idx_params:
            return idx_params, "index_json"

        # 2) sum shard bytes
        bytes_total = self._sum_shard_file_sizes(repo_id, siblings or [])
        if bytes_total:
            # Try to detect precision from filenames/metadata to convert bytes -> params
            # Heuristic: if any filename contains 'fp16' or 'float16' -> 2 bytes/param
            precision = "unknown"
            # quick filename check
            joined = " ".join(siblings).lower() if siblings else ""
            if "fp16" in joined or "float16" in joined or "bf16" in joined:
                precision = "fp16/bf16"
                bytes_per_param = 2
            elif "fp8" in joined:
                precision = "fp8"
                # fp8 storage can be 1 byte or packed; assume ~1 byte/param for storage but params semantics differ
                bytes_per_param = 1
            elif "int8" in joined or "int4" in joined or "gptq" in joined:
                precision = "int"
                bytes_per_param = 1
            else:
                # If config suggests dtype
                cfg_dtype = None
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
                # default conservative guess: 2 bytes/param (fp16)
                if not cfg_dtype:
                    bytes_per_param = 2
            try:
                approx_params = int(bytes_total // bytes_per_param)
                return approx_params, f"shard_size_sum ({precision})"
            except Exception:
                logger.debug("Failed bytes->params conversion")
                return None, "shard_size_sum_failed"

        # 3) fallback: config heuristics (very rough)
        if config:
            try:
                h = int(safe_get(config, "hidden_size") or safe_get(config, "d_model") or 0)
                l = int(safe_get(config, "num_hidden_layers") or safe_get(config, "n_layer") or 0)
                v = int(safe_get(config, "vocab_size") or safe_get(config, "n_vocab") or 0)
                if h and l:
                    # coarse transformer estimate
                    approx = v * h + l * (h * h * 12)
                    return int(approx), "config_heuristic"
            except Exception:
                pass

        return None, "unknown"

    # ---- Quantization / precision detection ----
    def detect_quant_and_precision(self, repo_id: str, config: Optional[Dict], siblings: List[str]) -> Dict[str, Any]:
        """
        Returns dictionary:
          { "quantized": bool, "quant_methods": [...], "precision": "fp16|bf16|fp8|int8|unknown" }
        Rules:
          - If quantization_config.json exists -> quantized True
          - Else look at filenames and index for dtype hints -> set precision
          - If only fp8 present -> precision=fp8 but quantized=False (fp8 is precision)
        """
        result = {"quantized": False, "quant_methods": [], "precision": "unknown"}

        # check quantization config
        qconf = self.load_json_quiet(repo_id, "quantization_config.json")
        if qconf:
            result["quantized"] = True
            # try to extract method
            m = qconf.get("method") or qconf.get("quantization_method")
            if m:
                result["quant_methods"].append(m)
            else:
                result["quant_methods"].append("unknown")

        # scan filenames
        joined = " ".join(siblings).lower() if siblings else ""
        if "gptq" in joined:
            result["quantized"] = True
            result["quant_methods"].append("gptq")
        if "bnb" in joined or "bitsandbytes" in joined or "bitsandbytes" in joined:
            result["quantized"] = True
            result["quant_methods"].append("bitsandbytes")
        if "awq" in joined:
            result["quantized"] = True
            result["quant_methods"].append("awq")

        # detect precision by filename hints
        if "fp16" in joined or "float16" in joined:
            result["precision"] = "fp16"
        elif "bf16" in joined:
            result["precision"] = "bf16"
        elif "fp8" in joined:
            result["precision"] = "fp8"
        elif "int8" in joined:
            result["precision"] = "int8"
        elif "int4" in joined:
            result["precision"] = "int4"

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

    def load_json_quiet(self, repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
        try:
            return self.load_json(repo_id, filename)
        except Exception:
            return None

    # ---- Tokenizer analysis ----
    def analyze_tokenizer(self, tokenizer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not tokenizer:
            return {"present": False}
        # Try best-effort extraction from tokenizer.json or tokenizer_config.json
        info = {"present": True, "type": None, "vocab_size": None, "model_max_length": None, "special_tokens": [], "truncation": None, "normalization": None, "lowercase": None}
        # tokenizer.json structure varies: 'model' -> {type, vocab}, or tokenizer_config has metadata
        model_part = tokenizer.get("model") if isinstance(tokenizer, dict) else None
        if model_part:
            info["type"] = model_part.get("type") or model_part.get("model_type") or info["type"]
            # vocab size might be a dict or list length
            vocab = model_part.get("vocab")
            if isinstance(vocab, dict):
                info["vocab_size"] = len(vocab)
            elif isinstance(vocab, list):
                info["vocab_size"] = len(vocab)
        # tokenizer_config.json style keys
        if "tokenizer_class" in tokenizer:
            info["type"] = tokenizer.get("tokenizer_class")
        if "model_max_length" in tokenizer:
            info["model_max_length"] = tokenizer.get("model_max_length")
        if "truncation" in tokenizer:
            info["truncation"] = tokenizer.get("truncation")
        if "do_lower_case" in tokenizer:
            info["lowercase"] = bool(tokenizer.get("do_lower_case"))
        # special tokens:
        # tokenizer.json may have 'added_tokens' or 'added_tokens_decoder'
        at = tokenizer.get("added_tokens") or tokenizer.get("added_tokens_decoder") or tokenizer.get("special_tokens_map") or {}
        if isinstance(at, dict):
            info["special_tokens"] = list(at.keys())
        elif isinstance(at, list):
            # list of dicts maybe
            toks = []
            for t in at:
                if isinstance(t, dict):
                    if "content" in t:
                        toks.append(t["content"])
                    elif "token" in t:
                        toks.append(t["token"])
            info["special_tokens"] = toks
        # Some tokenizer.json include normalizer info
        if "normalizer" in tokenizer:
            info["normalization"] = tokenizer.get("normalizer")
        return info

    # ---- Extra architecture extraction ----
    def extract_architecture_extras(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        extras = {}
        if not config:
            return extras
        keys = {
            "intermediate_size": "intermediate_size",
            "hidden_dropout_prob": "hidden_dropout_prob",
            "attention_probs_dropout_prob": "attention_probs_dropout_prob",
            "layer_norm_eps": "layer_norm_eps",
            "activation_function": "activation_function",
            "rope_theta": "rope_theta",
            "rope_scaling": "rope_scaling",
            "sliding_window": "sliding_window",
            "use_cache": "use_cache",
            "tie_word_embeddings": "tie_word_embeddings",
            "num_key_value_heads": "num_key_value_heads",
            "kv_head_dim": "kv_head_dim",
        }
        for k, out in keys.items():
            if k in config:
                extras[out] = config[k]
        # layer norm type inference (RMSNorm vs LayerNorm)
        if config.get("rms_norm", False) or "rms" in str(config.get("norm_type", "")).lower():
            extras["norm_type"] = "RMSNorm"
        elif "layer_norm" in str(config.get("norm_type", "")).lower() or "layernorm" in str(config.get("norm_type", "")).lower():
            extras["norm_type"] = "LayerNorm"
        else:
            # look for layer_norm_eps presence
            if "layer_norm_eps" in config:
                extras["norm_type"] = "LayerNorm"
        return extras

    # ---- Build report & warnings ----
    def build_report(self, repo_id: str) -> Dict[str, Any]:
        logger.info(f"Inspecting {repo_id} ...")
        if not repo_exists(repo_id, token=self.token):
            raise ValueError(f"Repository {repo_id} not found or not accessible")

        model_info = self.fetch_model_info(repo_id) or {}
        config = self.load_json(repo_id, "config.json")
        # also try tokenizer_config.json if tokenizer.json missing
        tokenizer = self.load_json(repo_id, "tokenizer.json") or self.load_json(repo_id, "tokenizer_config.json")
        siblings = model_info.get("siblings", []) if model_info else []

        # license
        license_info = self._get_license_from_readme(repo_id)

        # param count
        param_count, param_method = self.estimate_param_count(repo_id, config, siblings)

        # quant and precision
        quant_info = self.detect_quant_and_precision(repo_id, config, siblings)

        # tokenizer analysis
        tokenizer_info = self.analyze_tokenizer(tokenizer)

        # architecture extras
        arch_extras = self.extract_architecture_extras(config)

        # model type detection & custom architecture handling
        model_type_raw = None
        if config:
            model_type_raw = config.get("model_type") or (config.get("architectures")[0] if config.get("architectures") else None)
        model_type = model_type_raw or "unknown"
        if isinstance(model_type, list):
            model_type = model_type[0]
        model_type_lower = str(model_type).lower() if model_type else "unknown"
        if model_type_lower not in SUPPORTED_TRANSFORMERS:
            model_type_label = f"{model_type} (custom)"
        else:
            model_type_label = model_type

        # warnings
        warnings = []
        if config is None:
            warnings.append("Missing config.json")
        if tokenizer is None:
            warnings.append("Missing tokenizer.json / tokenizer_config.json")
        if license_info == "Unknown":
            warnings.append("License not detected in README")
        if not any(s.lower().endswith(".safetensors") for s in siblings):
            warnings.append("No SafeTensors weight files detected — verify weight safety and integrity")
        # huge model warning
        if param_count and param_count >= 500_000_000_000:
            warnings.append("Ultra-large model detected (>=500B parameters). Many estimates may be unreliable; prefer shard index metadata.")

        # assemble structured report
        report = {
            "repo_id": repo_id,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "model_info": model_info,
            "license": license_info,
            "config": config,
            "tokenizer": tokenizer_info,
            "model_type": model_type_label,
            "param_count": param_count,
            "param_count_method": param_method,
            "param_count_human": humanize_params(param_count),
            "quant_info": quant_info,
            "arch_extras": arch_extras,
            "file_analysis": self._analyze_files(siblings),
            "warnings": warnings,
        }
        return report

    # ---- file analysis helper reused from earlier logic ----
    def _analyze_files(self, siblings: List[str]) -> Dict[str, Any]:
        file_mappings = {
            'safetensors': ('.safetensors', 'SafeTensors'),
            'pytorch': (('.bin', '.pt', '.pth'), 'PyTorch'),
            'gguf': ('.gguf', 'GGUF'),
            'onnx': ('.onnx', 'ONNX')
        }
        analysis = {
            'total_files': len(siblings),
            'formats': set(),
            'model_files': [],
            'config_files': [],
            'other_files': []
        }
        for format_key in file_mappings:
            analysis[f'has_{format_key}'] = False
        config_file_names = {
            'config.json', 'tokenizer.json', 'tokenizer_config.json',
            'generation_config.json', 'model.safetensors.index.json'
        }
        for file in siblings:
            file_lower = file.lower()
            categorized = False
            for format_key, (extensions, format_name) in file_mappings.items():
                extensions = extensions if isinstance(extensions, tuple) else (extensions,)
                if any(file_lower.endswith(ext) for ext in extensions):
                    analysis[f'has_{format_key}'] = True
                    analysis['formats'].add(format_name)
                    analysis['model_files'].append(file)
                    categorized = True
                    break
            if not categorized:
                if file_lower in config_file_names:
                    analysis['config_files'].append(file)
                else:
                    analysis['other_files'].append(file)
        analysis['formats'] = list(analysis['formats'])
        return analysis

    def _get_license_from_readme(self, repo_id: str) -> str:
        try:
            path = hf_hub_download(repo_id=repo_id, filename="README.md", token=self.token)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # try to parse YAML front matter license
            if content.startswith("---"):
                parts = content.split("---")
                if len(parts) > 1:
                    # frontmatter in parts[1]
                    fm = parts[1]
                    for line in fm.splitlines():
                        if line.strip().lower().startswith("license:"):
                            return line.split(":", 1)[1].strip()
            # fallback: search for "license" mentions
            lowered = content.lower()
            if "license" in lowered:
                # try to find common license tokens
                for lic in ["mit", "apache-2.0", "apache-2", "apache2", "gpl", "bsd"]:
                    if lic in lowered:
                        return lic
            return "Unknown"
        except Exception:
            return "Unknown"

    # ---- Output helpers ----
    def format_markdown(self, rep: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("# 🤗 Model Inspector Report")
        lines.append(f"**Repository:** `{rep['repo_id']}`")
        lines.append(f"**Generated:** {rep['generated_at']}")
        lines.append("")
        mi = rep.get("model_info", {})
        # repo stats
        lines.append("## 📊 Repository Stats")
        lines.append(f"- **Author:** {mi.get('author','N/A')}")
        lines.append(f"- **License:** {rep.get('license','Unknown')}")
        lines.append(f"- **Gated:** {'Yes' if mi.get('gated') else 'No'}")
        lines.append(f"- **Private:** {'Yes' if mi.get('private') else 'No'}")
        lines.append(f"- **Library:** {mi.get('library_name','N/A')}")
        lines.append(f"- **Pipeline Tag:** {mi.get('pipeline_tag','N/A')}")
        lines.append(f"- **Likes:** {mi.get('likes',0):,}")
        lines.append(f"- **Downloads:** {mi.get('downloads',0):,}")
        created = mi.get("createdAt") or mi.get("created_at") or None
        last_mod = mi.get("lastModified") or mi.get("last_modified") or None
        if created:
            try:
                created_str = created.strftime("%Y-%m-%d")
            except Exception:
                created_str = str(created)
            lines.append(f"- **Created:** {created_str}")
        if last_mod:
            try:
                last_str = last_mod.strftime("%Y-%m-%d")
            except Exception:
                last_str = str(last_mod)
            lines.append(f"- **Last Modified:** {last_str}")
        lines.append("")

        # tags
        tags = mi.get("tags", []) or []
        if tags:
            tags_str = ", ".join(f"`{t}`" for t in tags[:20])
            lines.append("## 🏷️ Tags")
            lines.append(tags_str)
            lines.append("")

        # model architecture basics
        cfg = rep.get("config") or {}
        lines.append("## 🏗️ Model Architecture")
        # prefer explicit fields with safe access
        def field(cfg, *names):
            for n in names:
                if n in cfg and cfg[n] is not None:
                    return cfg[n]
            return None

        model_type_label = rep.get("model_type", "unknown")
        lines.append(f"- **Model Type:** `{model_type_label}`")
        hidden = field(cfg, "hidden_size", "d_model", "n_embd")
        layers = field(cfg, "num_hidden_layers", "n_layer", "num_layers")
        heads = field(cfg, "num_attention_heads", "n_head")
        vocab = field(cfg, "vocab_size", "n_vocab")
        lines.append(f"- **Hidden Size:** {hidden if hidden is not None else 'N/A'}")
        lines.append(f"- **Number of Layers:** {layers if layers is not None else 'N/A'}")
        lines.append(f"- **Attention Heads:** {heads if heads is not None else 'N/A'}")
        lines.append(f"- **Vocabulary Size:** {vocab if vocab is not None else 'N/A'}")
        lines.append("")
        # extras
        extras = rep.get("arch_extras", {})
        for k, v in extras.items():
            lines.append(f"- **{k.replace('_',' ').title()}:** {v}")
        lines.append("")
        # params & quant
        lines.append(f"- **Estimated Parameters:** {rep.get('param_count_human') } ({rep.get('param_count', 'N/A')})")
        q = rep.get("quant_info", {})
        quant_str = "✅ Detected" if q.get("quantized") else "❌ Not detected"
        methods = ", ".join(q.get("quant_methods", [])) or "n/a"
        precision = q.get("precision", "unknown")
        lines.append(f"- **Quantization:** {quant_str} ({methods})")
        lines.append(f"- **Precision / dtype hint:** {precision}")
        lines.append("")

        # tokenizer summary
        tok = rep.get("tokenizer", {})
        lines.append("## 📝 Tokenizer")
        if not tok or not tok.get("present", False):
            lines.append("⚠️ No tokenizer.json/tokenizer_config.json found")
        else:
            lines.append(f"- **Tokenizer Type:** {tok.get('type','N/A')}")
            lines.append(f"- **Tokenizer Vocab Size:** {tok.get('vocab_size','N/A')}")
            lines.append(f"- **Model Max Length:** {tok.get('model_max_length','N/A')}")
            stoks = tok.get("special_tokens") or []
            lines.append(f"- **Special Tokens:** {', '.join(stoks[:20]) if stoks else 'None detected'}")
            lines.append(f"- **Truncation:** {tok.get('truncation') or 'Not specified'}")
            if tok.get("lowercase") is not None:
                lines.append(f"- **Lowercasing:** {'Enabled' if tok.get('lowercase') else 'Disabled'}")
            if tok.get("normalization"):
                lines.append(f"- **Normalization:** present")
        lines.append("")

        # files
        fa = rep.get("file_analysis", {})
        lines.append("## 📁 Repository Files")
        lines.append(f"- **Total Files:** {fa.get('total_files', 0)}")
        lines.append(f"- **Model File Formats:** {', '.join(fa.get('formats', [])) if fa.get('formats') else 'None detected'}")
        for name, key in [("SafeTensors", "has_safetensors"), ("PyTorch", "has_pytorch"), ("GGUF", "has_gguf"), ("ONNX", "has_onnx")]:
            lines.append(f"- **{name}:** {'✅' if fa.get(key) else '❌'}")
        lines.append("")
        if fa.get("model_files"):
            lines.append("### Model Files")
            lines.append("```")
            display_files = sorted(fa["model_files"])[:50]
            lines.extend(display_files)
            if len(fa["model_files"]) > 50:
                lines.append(f"... and {len(fa['model_files'])-50} more")
            lines.append("```")
            lines.append("")

        # warnings
        warns = rep.get("warnings", [])
        if warns:
            lines.append("## ⚠️ Warnings & Notes")
            for w in warns:
                lines.append(f"- {w}")
            lines.append("")

        return "\n".join(lines)

    # ---- Save outputs ----
    def save_outputs(self, report: Dict[str, Any], md_path: Path = Path("model_inspection_report.md")) -> None:
        md = self.format_markdown(report)
        md_path.write_text(md, encoding="utf-8")
        logger.info(f"Saved report markdown to {md_path.resolve()}")



# -------------------------
# CLI / Entrypoint
# -------------------------
def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="HF Model Inspector (robust for large/custom models)")
    parser.add_argument("--repo-id", "-r", help="Hugging Face repo id (e.g., openai/gpt-oss-20b)", required=False)
    parser.add_argument("--hf-token", help="Hugging Face token (or set HF_TOKEN in env/.env)", required=False)
    parser.add_argument("--output-md", help="Path to markdown output", default="model_inspection_report.md")
    parser.add_argument("--output-json", help="Path to JSON output", default="model_inspector.json")
    args = parser.parse_args()

    repo_id = args.repo_id or os.getenv("INPUT_REPO_ID") or os.getenv("REPO_ID")
    hf_token = args.hf_token or os.getenv("INPUT_HF_TOKEN") or os.getenv("HF_TOKEN")

    if not repo_id:
        repo_id = input("Enter Hugging Face repo id (e.g. owner/model): ").strip()
        if not repo_id:
            logger.error("No repo id provided. Exiting.")
            sys.exit(2)

    inspector = ModelInspector(token=hf_token)

    try:
        report = inspector.build_report(repo_id)
        inspector.save_outputs(report, md_path=Path(args.output_md))

        # if running in GH Actions, set outputs
        if os.getenv("GITHUB_ACTIONS") and os.getenv("GITHUB_OUTPUT"):
            try:
                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                    f.write(f"report-path={Path(args.output_md).resolve()}\n")
                    f.write(f"report-json={Path(args.output_json).resolve()}\n")
                    # add a couple more outputs
                    f.write(f"param-count={report.get('param_count') or ''}\n")
                    f.write(f"gated={report.get('model_info', {}).get('gated', False)}\n")
            except Exception as e:
                logger.debug(f"Could not set GitHub outputs: {e}")

        print("\n" + "=" * 80)
        print(f"Report written to {Path(args.output_md).resolve()}")
        print("=" * 80)
    except Exception as e:
        logger.error(f"Inspection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
