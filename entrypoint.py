#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import requests
from huggingface_hub import (
    hf_hub_download, 
    login, 
    model_info,
    list_repo_files,
    HfApi,
    repo_exists
)
from dotenv import load_dotenv


def safe_request_hf_api(api: HfApi, repo_id: str, token: str = None) -> Optional[Dict]:
    """Safely request model info from HF API with error handling."""
    try:
        info = api.model_info(repo_id, token=token)
        return {
            'downloads': getattr(info, 'downloads', 0),
            'likes': getattr(info, 'likes', 0),
            'tags': getattr(info, 'tags', []),
            'pipeline_tag': getattr(info, 'pipeline_tag', None),
            'library_name': getattr(info, 'library_name', None),
            'created_at': getattr(info, 'created_at', None),
            'last_modified': getattr(info, 'last_modified', None),
            'private': getattr(info, 'private', False),
            'gated': getattr(info, 'gated', False),
            'disabled': getattr(info, 'disabled', False),
            'author': getattr(info, 'author', None),
            'sha': getattr(info, 'sha', None),
            'siblings': [s.rfilename for s in getattr(info, 'siblings', [])],
        }
    except Exception as e:
        print(f"⚠️ Could not fetch model info from API: {e}")
        return None


def load_json(repo_id: str, filename: str, token: str = None) -> Optional[Dict]:
    """Download and load a JSON file from a Hugging Face repo."""
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load {filename}: {e}")
        return None


def check_file_exists(repo_id: str, filename: str, token: str = None) -> bool:
    """Check if a file exists in the repo without downloading."""
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id, token=token)
        return filename in files
    except Exception:
        return False


def estimate_params(config: Dict) -> Optional[int]:
    """Enhanced parameter estimation for various model architectures."""
    try:
        model_type = config.get("model_type", "").lower()
        hidden_size = config.get("hidden_size") or config.get("d_model")
        num_layers = config.get("num_hidden_layers") or config.get("n_layers") or config.get("num_layers")
        vocab_size = config.get("vocab_size")
        
        if not all([hidden_size, num_layers, vocab_size]):
            return None
        
        # Different estimation formulas for different architectures
        if model_type in ["llama", "mistral", "qwen", "phi"]:
            # Transformer with RMSNorm, SwiGLU, RoPE
            intermediate_size = config.get("intermediate_size", hidden_size * 4)
            num_attention_heads = config.get("num_attention_heads", 32)
            
            # Embedding + output projection
            embed_params = vocab_size * hidden_size * 2
            
            # Per layer: attention + MLP + norms
            attn_params = hidden_size * hidden_size * 4  # Q, K, V, O projections
            mlp_params = hidden_size * intermediate_size * 2  # up and down projections
            norm_params = hidden_size * 2  # RMSNorm weights
            
            layer_params = attn_params + mlp_params + norm_params
            total_params = embed_params + (layer_params * num_layers)
            
        elif model_type in ["gpt2", "gpt_neox"]:
            # Standard transformer
            total_params = (
                vocab_size * hidden_size +  # embeddings
                num_layers * (
                    4 * hidden_size * hidden_size +  # attention
                    8 * hidden_size * hidden_size +  # MLP
                    2 * hidden_size  # layer norms
                ) +
                vocab_size * hidden_size  # output projection
            )
        else:
            # Generic estimation
            total_params = (
                vocab_size * hidden_size * 2 +  # embeddings + output
                num_layers * hidden_size * hidden_size * 12  # rough approximation
            )
        
        return int(total_params)
    except Exception as e:
        print(f"⚠️ Parameter estimation failed: {e}")
        return None


def format_size(size_bytes: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def analyze_model_files(siblings: List[str]) -> Dict[str, Any]:
    """Analyze the model files to determine format, size, and other characteristics."""
    analysis = {
        'total_files': len(siblings),
        'formats': set(),
        'has_safetensors': False,
        'has_pytorch': False,
        'has_gguf': False,
        'has_onnx': False,
        'config_files': [],
        'model_files': [],
        'other_files': []
    }
    
    for file in siblings:
        file_lower = file.lower()
        
        # Detect formats
        if file_lower.endswith('.safetensors'):
            analysis['has_safetensors'] = True
            analysis['formats'].add('SafeTensors')
            analysis['model_files'].append(file)
        elif file_lower.endswith(('.bin', '.pt', '.pth')):
            analysis['has_pytorch'] = True
            analysis['formats'].add('PyTorch')
            analysis['model_files'].append(file)
        elif file_lower.endswith('.gguf'):
            analysis['has_gguf'] = True
            analysis['formats'].add('GGUF')
            analysis['model_files'].append(file)
        elif file_lower.endswith('.onnx'):
            analysis['has_onnx'] = True
            analysis['formats'].add('ONNX')
            analysis['model_files'].append(file)
        elif file_lower in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 
                           'generation_config.json', 'model.safetensors.index.json']:
            analysis['config_files'].append(file)
        else:
            analysis['other_files'].append(file)
    
    analysis['formats'] = list(analysis['formats'])
    return analysis


def get_license_info(repo_id: str, token: str = None) -> Optional[str]:
    """Try to get license information from the model card or repo."""
    try:
        # Try to download README.md or model card
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", token=token)
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Look for license in YAML frontmatter or content
        if "license:" in content.lower():
            lines = content.split('\n')
            for line in lines:
                if line.strip().lower().startswith('license:'):
                    return line.split(':', 1)[1].strip()
        
        return "Unknown"
    except Exception:
        return "Unknown"


def format_summary(repo_id: str, config: Dict, tokenizer: Dict, 
                  model_info: Dict, file_analysis: Dict, license_info: str) -> str:
    """Format a comprehensive Markdown summary of model details."""
    lines = [
        f"# 🤗 Model Inspector Report",
        f"**Repository:** `{repo_id}`",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ""
    ]

    # Basic Model Information
    if model_info:
        lines.extend([
            "## 📊 Repository Stats",
            f"- **Downloads:** {model_info.get('downloads', 0):,}",
            f"- **Likes:** {model_info.get('likes', 0):,}",
            f"- **Author:** {model_info.get('author', 'N/A')}",
            f"- **Private:** {'Yes' if model_info.get('private') else 'No'}",
            f"- **Gated:** {'Yes' if model_info.get('gated') else 'No'}",
            f"- **Library:** {model_info.get('library_name', 'N/A')}",
            f"- **Pipeline Tag:** {model_info.get('pipeline_tag', 'N/A')}",
            f"- **License:** {license_info}",
            ""
        ])

        if model_info.get('created_at'):
            lines.append(f"- **Created:** {model_info['created_at'].strftime('%Y-%m-%d')}")
        if model_info.get('last_modified'):
            lines.append(f"- **Last Modified:** {model_info['last_modified'].strftime('%Y-%m-%d')}")
        
        lines.append("")

        # Tags
        if model_info.get('tags'):
            tags_str = ", ".join([f"`{tag}`" for tag in model_info['tags'][:10]])  # Limit to first 10
            lines.extend([
                "## 🏷️ Tags",
                tags_str,
                ""
            ])

    # Model Architecture
    if config:
        lines.extend([
            "## 🏗️ Model Architecture",
            f"- **Model Type:** `{config.get('model_type', 'N/A')}`",
            f"- **Hidden Size:** {config.get('hidden_size', 'N/A'):,}" if config.get('hidden_size') else "- **Hidden Size:** N/A",
            f"- **Number of Layers:** {config.get('num_hidden_layers', 'N/A')}" if config.get('num_hidden_layers') else "- **Number of Layers:** N/A",
            f"- **Attention Heads:** {config.get('num_attention_heads', 'N/A')}" if config.get('num_attention_heads') else "- **Attention Heads:** N/A",
            f"- **Vocabulary Size:** {config.get('vocab_size', 'N/A'):,}" if config.get('vocab_size') else "- **Vocabulary Size:** N/A",
            ""
        ])

        # Additional architecture details
        additional_fields = [
            ('intermediate_size', 'Intermediate Size'),
            ('max_position_embeddings', 'Max Position Embeddings'), 
            ('num_key_value_heads', 'Key-Value Heads'),
            ('rope_theta', 'RoPE Theta'),
            ('sliding_window', 'Sliding Window'),
            ('attention_dropout', 'Attention Dropout'),
            ('hidden_dropout', 'Hidden Dropout'),
            ('rope_scaling', 'RoPE Scaling')
        ]

        extra_details = []
        for field, label in additional_fields:
            if field in config:
                value = config[field]
                if isinstance(value, (int, float)):
                    if isinstance(value, int) and value > 1000:
                        extra_details.append(f"- **{label}:** {value:,}")
                    else:
                        extra_details.append(f"- **{label}:** {value}")
                else:
                    extra_details.append(f"- **{label}:** {value}")
        
        if extra_details:
            lines.extend(extra_details + [""])

        # Parameter estimation
        params = estimate_params(config)
        if params:
            if params >= 1_000_000_000:
                param_str = f"{params / 1_000_000_000:.1f}B"
            elif params >= 1_000_000:
                param_str = f"{params / 1_000_000:.1f}M"
            else:
                param_str = f"{params:,}"
            lines.extend([
                f"- **Estimated Parameters:** ~{param_str} ({params:,})",
                ""
            ])

        # Quantization detection
        quant_indicators = ['bits', 'quant', 'int4', 'int8', 'fp16', 'bf16']
        is_quantized = any(
            any(indicator in str(v).lower() for indicator in quant_indicators)
            for v in config.values()
        ) or any(indicator in config.get('model_type', '').lower() for indicator in quant_indicators)
        
        lines.extend([
            f"- **Quantization:** {'✅ Detected' if is_quantized else '❌ Not detected'}",
            ""
        ])
    else:
        lines.extend([
            "## ⚠️ Model Architecture",
            "Could not load `config.json`",
            ""
        ])

    # Tokenizer information
    if tokenizer:
        lines.extend([
            "## 📝 Tokenizer",
            f"- **Model Max Length:** {tokenizer.get('model_max_length', 'N/A'):,}" if isinstance(tokenizer.get('model_max_length'), int) else f"- **Model Max Length:** {tokenizer.get('model_max_length', 'N/A')}",
            f"- **Tokenizer Vocab Size:** {len(tokenizer.get('model', {}).get('vocab', {})) if tokenizer.get('model', {}).get('vocab') else 'N/A'}",
            ""
        ])

    # File analysis
    lines.extend([
        "## 📁 Repository Files",
        f"- **Total Files:** {file_analysis['total_files']}",
        f"- **Model File Formats:** {', '.join(file_analysis['formats']) if file_analysis['formats'] else 'None detected'}",
        f"- **SafeTensors:** {'✅' if file_analysis['has_safetensors'] else '❌'}",
        f"- **PyTorch:** {'✅' if file_analysis['has_pytorch'] else '❌'}",
        f"- **GGUF:** {'✅' if file_analysis['has_gguf'] else '❌'}",
        f"- **ONNX:** {'✅' if file_analysis['has_onnx'] else '❌'}",
        ""
    ])

    # File breakdown
    if file_analysis['model_files']:
        lines.extend([
            "### Model Files",
            "```"
        ])
        for file in sorted(file_analysis['model_files'])[:20]:  # Limit to first 20
            lines.append(file)
        if len(file_analysis['model_files']) > 20:
            lines.append(f"... and {len(file_analysis['model_files']) - 20} more")
        lines.extend(["```", ""])

    return "\n".join(lines)


def main():
    """Main execution function for GitHub Actions compatibility."""
    # Load environment variables
    load_dotenv()

    # Detect execution context
    repo_id = os.getenv("INPUT_REPO_ID")
    hf_token = os.getenv("INPUT_HF_TOKEN") or os.getenv("HF_TOKEN")

    if not repo_id:
        parser = argparse.ArgumentParser(description="Enhanced Hugging Face Model Inspector")
        parser.add_argument("--repo-id", required=True, help="Hugging Face repo ID (e.g., microsoft/DialoGPT-large)")
        parser.add_argument("--hf-token", help="Hugging Face access token (for gated/private repos)")
        args = parser.parse_args()
        repo_id = args.repo_id
        hf_token = hf_token or args.hf_token

    print(f"🔍 Inspecting model: {repo_id}")

    # Authenticate if token provided
    api = HfApi()
    if hf_token:
        print("🔐 Using Hugging Face token for authentication")
        login(token=hf_token, add_to_git_credential=False)

    try:
        # Check if repo exists and is accessible
        if not repo_exists(repo_id, token=hf_token):
            print(f"❌ Repository {repo_id} not found or not accessible")
            exit(1)

        # Gather all information
        print("📥 Fetching model information...")
        model_info = safe_request_hf_api(api, repo_id, hf_token)
        
        print("📥 Loading configuration files...")
        config = load_json(repo_id, "config.json", token=hf_token)
        tokenizer = load_json(repo_id, "tokenizer.json", token=hf_token)
        
        print("📥 Analyzing repository files...")
        siblings = model_info.get('siblings', []) if model_info else []
        file_analysis = analyze_model_files(siblings)
        
        print("📥 Fetching license information...")
        license_info = get_license_info(repo_id, hf_token)

        # Generate comprehensive summary
        summary = format_summary(repo_id, config, tokenizer, model_info, file_analysis, license_info)
        print("\n" + "="*80)
        print(summary)
        print("="*80)

        # Save to artifact
        out_path = Path("model_inspection_report.md")
        out_path.write_text(summary, encoding="utf-8")
        print(f"\n✅ Comprehensive report saved to {out_path.resolve()}")

        # Set GitHub Actions output if running in GA
        if os.getenv("GITHUB_ACTIONS"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"report-path={out_path}\n")
                if model_info:
                    f.write(f"downloads={model_info.get('downloads', 0)}\n")
                    f.write(f"likes={model_info.get('likes', 0)}\n")
                    f.write(f"gated={'true' if model_info.get('gated') else 'false'}\n")

    except Exception as e:
        print(f"❌ Error during inspection: {e}")
        exit(1)


# Entry point for GitHub Actions - no if __name__ == "__main__" needed
main()