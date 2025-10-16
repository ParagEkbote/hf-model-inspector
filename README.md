# HF Model Inspector

**A package and GitHub Action to generate quality reports on Hugging Face model stats and usage, addressing ambiguity in open model releases.**

---

## Overview

`hf_model_inspector` is a Python package and GitHub Action designed to provide **clear, structured reports** for models hosted on [Hugging Face](https://huggingface.co). Open model releases often come with incomplete or inconsistent metadata, making it hard to quickly assess model size, architecture, quantization, and usage statistics.  

This tool helps you:  

- Inspect **model metadata** including architecture, parameters, and downloads.  
- Handle **quantization info** cleanly, even when formats differ across releases.  
- Generate **JSON and Markdown reports** for documentation or review purposes.  
- Recommend **suitable models for your GPU** based on memory constraints.  
- Automate reporting with a **GitHub Action** for CI/CD pipelines.  

---

## Installation

```bash
pip install hf_model_inspector
````

> Optional: For private models, you can use a Hugging Face token.

---

## Quick Start

### Example 1: Inspect a model and print summary

```python
from hf_model_inspector import get_model_report_json

repo_id = "openai/gpt-oss-20b"
report = get_model_report_json(repo_id)

total_params = report["parameters"]["total"]
param_str = f"{total_params:,}" if total_params else "Unknown"

quant_info = report.get("quantization", {})
if quant_info.get("quantized"):
    methods = ", ".join(quant_info.get("quant_methods", [])) or "Unknown"
    precision = quant_info.get("precision", "Unknown")
    quant_status = f"{methods} ({precision})"
else:
    quant_status = quant_info.get("dtype", "Not Quantized") or "Not Quantized"

print(f"Model: {report['repo_id']}")
print(f"Architecture: {report['architecture']}")
print(f"Parameters: {param_str}")
print(f"Quantization: {quant_status}")
print(f"Downloads: {report['metadata']['downloads']}")
print(f"Tags: {', '.join(report['metadata']['tags']) if report['metadata']['tags'] else 'None'}")
```

---

### Example 2: Full inspection and Markdown report

```python
from hf_model_inspector import get_model_report_md, save_model_report
from hf_model_inspector.loader import authenticate_hf

token = authenticate_hf()
repo_id = "openai/gpt-oss-20b"

# Generate and save Markdown report
report_md = get_model_report_md(repo_id, token)
save_model_report(repo_id, md_path="model_report.md", token=token)

print("Markdown report saved as 'model_report.md'")
```

---

## GitHub Action Integration

You can automate model reporting on every push or PR using our GitHub Action:

```yaml
name: HF Model Inspector

on:
  push:
    branches: [main]

jobs:
  inspect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run HF Model Inspector
        uses: ParagEkbote/hf-model-inspector@v1.0.0
        with:
          repo_id: "openai/gpt-oss-20b"
          token: ${{ secrets.HF_TOKEN }}
```

This will automatically generate and store **JSON/Markdown reports** for your chosen model.

---

## Features

* ✅ Inspect **public and private models**.
* ✅ Clean handling of **quantization** and **parameter counts**.
* ✅ Save **JSON or Markdown reports**.
* ✅ Recommend models suitable for your GPU.
* ✅ Automate with **GitHub Actions** for reproducible reporting.

---

