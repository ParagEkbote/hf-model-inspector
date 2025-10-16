"""Inspect a model and print its metadata summary as JSON."""
from hf_model_inspector import get_model_report_json

repo_id = "openai/gpt-oss-20b"
report = get_model_report_json(repo_id)

total_params = report["parameters"]["total"]
param_str = f"{total_params:,}" if total_params is not None else "Unknown"

# Enhanced quantization info extraction
quant_info = report.get("quantization", {})
if quant_info and quant_info.get("quantized"):
    quant_methods = quant_info.get("quant_methods", [])
    quant_methods_str = ", ".join(quant_methods) if quant_methods else "Unknown"
    precision = quant_info.get("precision", "Unknown")
    quant_status = f"{quant_methods_str} ({precision})"
else:
    # Fallback to old format if available
    quant_status = quant_info.get("dtype", "Not Quantized") if quant_info else "Not Quantized"

print("📘 Model Report Summary:")
print(f"Model ID: {report['repo_id']}")
print(f"Architecture: {report['architecture']}")
print(f"Total Params: {param_str}")
print(f"Quantization: {quant_status}")
print(f"Downloads: {report['metadata']['downloads']}")
print(f"Tags: {', '.join(report['metadata']['tags']) if report['metadata']['tags'] else 'None'}")

# Optional: Print detailed quantization info if available
if quant_info and quant_info.get("quantized"):
    print("\n🔧 Quantization Details:")
    print(f"  Methods: {', '.join(quant_info.get('quant_methods', []))}")
    print(f"  Precision: {quant_info.get('precision', 'Unknown')}")