import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def format_markdown(report: dict[str, Any]) -> str:
    """
    Format a model report dictionary as Markdown.

    Args:
        report: Dictionary containing model report data

    Returns:
        Formatted Markdown string
    """
    lines = []

    # Title
    repo_id = report.get("repo_id", "Unknown Model")
    lines.append(f"# Model Report: {repo_id}")
    lines.append("")

    # Basic Info
    lines.append("## Basic Information")
    lines.append("")
    lines.append(f"- **Repository**: {repo_id}")
    lines.append(f"- **Architecture**: {report.get('architecture', 'Unknown')}")
    lines.append(f"- **Model Type**: {report.get('model_type', 'Unknown')}")
    lines.append("")

    # Metadata
    metadata = report.get("metadata", {})
    if metadata:
        lines.append("## Metadata")
        lines.append("")
        lines.append(f"- **Downloads**: {metadata.get('downloads', 'N/A'):,}")
        lines.append(f"- **Likes**: {metadata.get('likes', 'N/A'):,}")
        lines.append(f"- **Library**: {metadata.get('library', 'N/A')}")
        lines.append(f"- **Pipeline Tag**: {metadata.get('pipeline_tag', 'N/A')}")

        tags = metadata.get("tags", [])
        if tags:
            lines.append(f"- **Tags**: {', '.join(tags[:10])}")
        lines.append("")

    # Parameters
    params = report.get("parameters", {})
    if params:
        lines.append("## Parameters")
        lines.append("")

        total = params.get("total")
        if total:
            millions = params.get("total_millions")
            billions = params.get("total_billions")
            lines.append(f"- **Total Parameters**: {total:,}")
            if millions:
                lines.append(f"  - {millions}M parameters")
            if billions and billions >= 1:
                lines.append(f"  - {billions}B parameters")
        else:
            lines.append("- **Total Parameters**: Unable to estimate")

        method = params.get("estimation_method")
        if method:
            lines.append(f"- **Estimation Method**: {method}")
        lines.append("")

    # Quantization
    quant = report.get("quantization", {})
    if quant:
        lines.append("## Quantization & Precision")
        lines.append("")

        is_quantized = quant.get("quantized", False)
        lines.append(f"- **Quantized**: {'Yes' if is_quantized else 'No'}")

        methods = quant.get("quant_methods", [])
        if methods:
            lines.append(f"- **Quantization Methods**: {', '.join(methods)}")

        precision = quant.get("precision", "unknown")
        lines.append(f"- **Precision**: {precision}")
        lines.append("")

    # Architecture Details
    arch = report.get("architecture_extras", {})
    if arch:
        lines.append("## Architecture Details")
        lines.append("")

        for key, value in arch.items():
            if value is not None:
                # Format key nicely
                formatted_key = key.replace("_", " ").title()
                lines.append(f"- **{formatted_key}**: {value}")
        lines.append("")

    # Tokenizer
    tokenizer = report.get("tokenizer", {})
    if tokenizer and tokenizer.get("present"):
        lines.append("## Tokenizer")
        lines.append("")

        tok_type = tokenizer.get("type") or tokenizer.get("tokenizer_class")
        if tok_type:
            lines.append(f"- **Type**: {tok_type}")

        vocab_size = tokenizer.get("vocab_size")
        if vocab_size:
            lines.append(f"- **Vocabulary Size**: {vocab_size:,}")

        max_length = tokenizer.get("model_max_length")
        if max_length and max_length != float("inf"):
            lines.append(f"- **Max Length**: {max_length}")

        special_tokens = tokenizer.get("special_tokens", [])
        if special_tokens:
            if isinstance(special_tokens, list):
                lines.append(
                    f"- **Special Tokens**: {', '.join(str(t) for t in special_tokens[:10])}"
                )
            else:
                lines.append(f"- **Special Tokens**: {special_tokens}")

        lines.append("")

    # Architecture Details (from config)
    arch_details = report.get("architecture_details", {})
    if arch_details and any(v is not None for v in arch_details.values()):
        lines.append("## Model Configuration")
        lines.append("")

        for key in [
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "intermediate_size",
            "max_position_embeddings",
            "vocab_size",
        ]:
            value = arch_details.get(key)
            if value is not None:
                formatted_key = key.replace("_", " ").title()
                lines.append(
                    f"- **{formatted_key}**: {value:,}"
                    if isinstance(value, int)
                    else f"- **{formatted_key}**: {value}"
                )

        hidden_act = arch_details.get("hidden_act")
        if hidden_act:
            lines.append(f"- **Hidden Activation**: {hidden_act}")

        lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Report generated by HF Model Inspector*")

    return "\n".join(lines)


def save_outputs(content: str, filepath: str) -> None:
    """
    Save content to a file.

    Args:
        content: String content to save
        filepath: Path where to save the file

    Raises:
        IOError: If file cannot be written
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Write the file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Successfully saved output to {filepath}")
    except Exception as err:
        logger.error(f"Failed to save output to {filepath}")
        raise OSError(f"Could not write to {filepath}") from err
