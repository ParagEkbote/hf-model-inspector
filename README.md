# hf-model-inspector

Here’s a polished `README.md` draft for your GitHub workflow based on your final `entrypoint.py` that produces a comprehensive model inspection report:

# HF Transformers Model Inspector

This repository provides a **GitHub Actions workflow** and a **Python CLI tool** (`entrypoint.py`) to inspect Hugging Face Transformers models. It generates a **comprehensive Markdown report** with key model details, including architecture, parameter counts, and more.

---

## Features

- Inspect any Hugging Face model by its repo ID.
- Generates a **detailed Markdown report** (`model_inspection_report.md`) containing:
  - Model architecture summary
  - Number of parameters
  - Layer-wise details
  - Any other relevant metadata
- Works both in **local Python environments** and via **Docker/GitHub Actions**.
- Simple integration with CI/CD pipelines for model validation.

---

## Requirements

- Python 3.10+ (3.12 recommended for CI workflow)
- `transformers`
- `huggingface_hub`
- `python-dotenv`


---

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

### 2. Run the Inspector

```bash
python entrypoint.py --repo-id <MODEL_REPO_ID>
```

Example:

```bash
python entrypoint.py --repo-id bert-base-uncased
```

The output will be saved as:

```
model_inspection_report.md
```

---

## GitHub Actions Workflow

The repository includes a CI workflow (`.github/workflows/ci.yml`) with two jobs:

### **1. Test Job**

* Checks out the code.
* Sets up Python (3.12).
* Installs dependencies from `requirements.txt`.
* Runs a **smoke test** using a sample model:

```yaml
python entrypoint.py --repo-id bert-base-uncased
```

### **2. Docker Job**

* Builds a Docker image containing `entrypoint.py` and dependencies.
* Runs the inspector inside the Docker container.
* Saves the Markdown report in the workspace for inspection.

Docker run example:

```bash
docker run --rm -v ${{ github.workspace }}:/app hf-model-inspector \
    python /app/entrypoint.py --repo-id moonshotai/Kimi-K2-Instruct-0905
```

---

## Output

The workflow generates a **Markdown report** (`model_inspection_report.md`) in your repository workspace. It includes:

* Model name and repo ID
* Architecture type
* Number of parameters
* Layer-level details
* Any additional metadata

## Environment Variables

For Hugging Face API access, set your token:

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
```

In GitHub Actions:

```yaml
env:
  HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

---

