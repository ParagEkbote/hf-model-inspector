import os
from hf_model_inspector import (
    get_model_report_json,
    get_model_report_md,
    save_model_report,
    get_lora_info,
    recommend_models_for_gpu,
)
from hf_model_inspector.loader import authenticate_hf, HFModelLoader

def main():
    # 1️⃣ Authenticate (optional token or cached)
    token = authenticate_hf()

    # 2️⃣ Initialize loader
    loader = HFModelLoader(token)

    # 3️⃣ Pick a test repo (replace with a real HF model)
    repo_id = "facebook/opt-125m"  # example public model

    # 4️⃣ Test fetch_model_info
    info = loader.fetch_model_info(repo_id)
    print("Model Info:", info)

    # 5️⃣ Test load_json_quiet
    config = loader.load_json_quiet(repo_id, "config.json")
    print("Config:", config)

    # 6️⃣ Test load_lora_info (likely None for this model)
    lora_info = loader.load_lora_info(repo_id)
    print("LoRA Info:", lora_info)

    # 7️⃣ Test get_model_report_json
    report_json = get_model_report_json(repo_id, token)
    print("Report JSON:", report_json)

    # 8️⃣ Test get_model_report_md
    report_md = get_model_report_md(repo_id, token)
    print("Report MD:\n", report_md)

    # 9️⃣ Test save_model_report
    save_model_report(repo_id, md_path="test_model_report.md", token=token)
    print("Markdown report saved as test_model_report.md")

    # 🔟 Test GPU recommendation
    gpu_specs = {"name": "RTX 3090", "memory_gb": 24}
    recommended = recommend_models_for_gpu(gpu_specs)
    print("Recommended models for GPU:", recommended)

if __name__ == "__main__":
    main()
