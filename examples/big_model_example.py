from hf_model_inspector import (
    get_model_report_json,
    get_model_report_md,
    save_model_report,
)
from hf_model_inspector.loader import authenticate_hf

def summarize_model(report: dict) -> None:
    """Print a clean summary of the model metadata."""
    total_params = report["parameters"].get("total")
    param_str = f"{total_params:,}" if total_params is not None else "Unknown"

    # Enhanced quantization info
    quant_info = report.get("quantization", {})
    if quant_info and quant_info.get("quantized"):
        quant_methods = quant_info.get("quant_methods", [])
        precision = quant_info.get("precision", "Unknown")
        quant_status = f"{', '.join(quant_methods) if quant_methods else 'Unknown'} ({precision})"
    else:
        quant_status = quant_info.get("dtype", "Not Quantized") if quant_info else "Not Quantized"

    print("📘 Model Report Summary:")
    print(f"Model ID: {report['repo_id']}")
    print(f"Architecture: {report['architecture']}")
    print(f"Total Params: {param_str}")
    print(f"Quantization: {quant_status}")
    print(f"Downloads: {report['metadata'].get('downloads', 'Unknown')}")
    tags = report['metadata'].get('tags')
    print(f"Tags: {', '.join(tags) if tags else 'None'}")

    # Optional: detailed quantization info
    if quant_info.get("quantized"):
        print("\n🔧 Quantization Details:")
        print(f"  Methods: {', '.join(quant_info.get('quant_methods', []))}")
        print(f"  Precision: {quant_info.get('precision', 'Unknown')}")

def main():
    repo_id = "moonshotai/Kimi-K2-Instruct-0905"  # Replace with your model
    print("=" * 60)
    print(f"Inspecting model: {repo_id}")
    print("=" * 60)

    # Optional: authenticate if needed
    token = authenticate_hf()  # Can be None for public models

    try:
        # Fetch JSON report
        report = get_model_report_json(repo_id, token)
        summarize_model(report)
    except Exception as e:
        print(f"✗ Failed to fetch report: {e}")
        return

    # Optional: generate and save Markdown report
    try:
        report_md = get_model_report_md(repo_id, token)
        save_model_report(repo_id, md_path="model_report.md", token=token)
        print("\n✓ Markdown report saved as 'model_report.md'")
    except Exception as e:
        print(f"⚠ Could not save Markdown report: {e}")

    print("\n✅ Inspection complete!")

if __name__ == "__main__":
    main()
