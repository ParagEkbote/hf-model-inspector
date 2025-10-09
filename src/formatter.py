# formatter.py
from typing import Dict
from pathlib import Path



def save_outputs(report: Dict[str, Any], md_path: Path = Path("model_inspection_report.md")) -> None:
    """Save report as Markdown file."""
    md = self.format_markdown(report)
    md_path.write_text(md, encoding="utf-8")
    logger.info(f"Saved report markdown to {md_path.resolve()}")
