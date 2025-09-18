#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, login
from dotenv import load_dotenv


def load_json(repo_id: str, filename: str, token: str = None):
    """Download and load a JSON file from a Hugging Face repo."""
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load {filename}: {e}")
        return None


def estimate_params(config):
    """Rough parameter estimation based on config.json."""
    try:
        hidden_size = config.get("hidden_size")
        num_layers = config.get("num_hidden_layers")
        vocab_size = config.get("vocab_size")
        if hidden_size and num_layers and vocab_size:
            return int(2 * hidden_size * hidden_size * num_layers + vocab_size * hidden_size)
    except Exception:
        return None
    return None


def format_summary(repo_id: str, config, tokenizer):
    """Format a Markdown summary of model details."""
    lines = [f"## 🤗 Model Inspector Report for `{repo_id}`\n"]

    if config:
        lines.append(f"**Model Type:** {config.get('model_type', 'N/A')}")
        lines.append(f"**Hidden Size:** {config.get('hidden_size', 'N/A')}")
        lines.append(f"**Num Layers:** {config.get('num_hidden_layers', 'N/A')}")
        lines.append(f"**Num Attention Heads:** {config.get('num_attention_heads', 'N/A')}")
        lines.append(f"**Max Position Embeddings:** {config.get('max_position_embeddings', 'N/A')}")
        lines.append(f"**Vocabulary Size:** {config.get('vocab_size', 'N/A')}")
        lines.append(f"**Transformers Version Required:** {config.get('transformers_version', 'N/A')}")

        params = estimate_params(config)
        if params:
            lines.append(f"**Estimated Params:** ~{params:,}")

        # Check quantization
        if any("quant" in str(v).lower() for v in config.values()):
            lines.append("**Quantization:** ✅ Applied")
        else:
            lines.append("**Quantization:** ❌ Not detected")
    else:
        lines.append("⚠️ Could not load `config.json`")

    if tokenizer:
        lines.append("\n### Tokenizer")
        lines.append(f"- Model Max Length: {tokenizer.get('model_max_length', 'N/A')}")
        lines.append(f"- Vocab Size: {tokenizer.get('vocab_size', 'N/A')}")

    return "\n".join(lines)


def main():
    # Load secrets from .env if available (local dev only)
    load_dotenv()

    # Detect mode: GitHub Actions or CLI
    repo_id = os.getenv("INPUT_REPO_ID")
    hf_token = os.getenv("INPUT_HF_TOKEN") or os.getenv("HF_TOKEN")

    if not repo_id:
        parser = argparse.ArgumentParser(description="Hugging Face Model Inspector")
        parser.add_argument("--repo-id", required=True, help="Hugging Face repo id (e.g., openai/gpt-oss-20b)")
        parser.add_argument("--hf-token", help="Hugging Face access token (for gated/private repos)")
        args = parser.parse_args()
        repo_id = args.repo_id
        hf_token = hf_token or args.hf_token

    if hf_token:
        print("🔐 Using Hugging Face token for authentication (token masked)")
        login(token=hf_token, add_to_git_credential=False)

    print(f"🔍 Inspecting model: {repo_id}")

    # Try loading config + tokenizer
    config = load_json(repo_id, "config.json", token=hf_token)
    tokenizer = load_json(repo_id, "tokenizer.json", token=hf_token)

    # Generate summary
    summary = format_summary(repo_id, config, tokenizer)
    print("\n" + summary)

    # Save to artifact
    out_path = Path("model_summary.md")
    out_path.write_text(summary, encoding="utf-8")
    print(f"\n✅ Summary written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
