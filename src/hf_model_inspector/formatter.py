from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def format_markdown(report: Dict[str, Any]) -> str:
    """Convert model inspection report into Markdown format."""
    if not report:
        return "# Model Inspection Report\n\n_No data available._"

    lines = ["# Model Inspection Report\n"]

    if "id" in report:
        lines.append(f"## 🧠 Model: `{report['id']}`\n")

    for key, value in report.items():
        if key == "lora_info" and isinstance(value, dict):
            lines.append("### ⚡ LoRA Adapter Info\n")
            for subkey, subval in value.items():
                lines.append(f"- **{subkey}**: {subval if subval is not None else 'N/A'}")
            lines.append("")
            continue

        if isinstance(value, dict):
            lines.append(f"### {key}\n")
            for subkey, subval in value.items():
                lines.append(f"- **{subkey}**: {subval}")
        elif isinstance(value, list):
            lines.append(f"### {key}\n" + "\n".join(f"- {v}" for v in value))
        else:
            lines.append(f"**{key}:** {value}")

        lines.append("")  # blank line between sections

    return "\n".join(lines)


def save_outputs(report: Dict[str, Any], md_path: Path = Path("model_inspection_report.md")) -> None:
    """Save the given report dictionary as a Markdown file."""
    try:
        md_content = format_markdown(report)
        md_path.write_text(md_content, encoding="utf-8")
        logger.info(f"✅ Saved report markdown to {md_path.resolve()}")
    except Exception as e:
        logger.error(f"❌ Failed to save report to {md_path}: {e}")
