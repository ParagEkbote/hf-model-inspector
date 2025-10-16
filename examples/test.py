from hf_model_inspector import (
    get_model_report_json,
    get_model_report_md,
    save_model_report,
    recommend_models_for_gpu,
)
from hf_model_inspector.loader import authenticate_hf, HFModelLoader


def main():
    print("=" * 60)
    print("HF Model Inspector Test Suite")
    print("=" * 60)

    # 1️⃣ Authenticate (optional token or cached)
    print("\n1. Authenticating...")
    token = authenticate_hf()
    print("✓ Authentication successful")

    # 2️⃣ Initialize loader
    print("\n2. Initializing loader...")
    loader = HFModelLoader(token)
    print("✓ Loader initialized")

    # 3️⃣ Pick a test repo (replace with a real HF model)
    repo_id = "torchao-testing/opt-125m-Float8DynamicActivationFloat8WeightConfig-v1-0.13.dev"  # public small model for testing
    print(f"\n3. Testing with model: {repo_id}")

    # 4️⃣ Test fetch_model_info
    print("\n4. Fetching model info...")
    info = loader.fetch_model_info(repo_id)
    if info:
        print(f"✓ Model: {info.get('id')}")
        print(f"  Downloads: {info.get('downloads')}")
        print(f"  Likes: {info.get('likes')}")
    else:
        print("✗ Failed to fetch model info")

    # 5️⃣ Test load_json_quiet
    print("\n5. Loading config.json...")
    config = loader.load_json_quiet(repo_id, "config.json")
    if config:
        print(f"✓ Config loaded: {list(config.keys())[:5]}...")
    else:
        print("✗ No config found")

    # 6️⃣ Test load_lora_info
    print("\n6. Checking for LoRA info...")
    lora_info = loader.load_lora_info(repo_id)
    if lora_info:
        print(f"✓ LoRA info: {lora_info}")
    else:
        print("✓ No LoRA info (expected for this model)")

    # 7️⃣ Test get_model_report_json
    print("\n7. Generating JSON report...")
    report_json = None
    try:
        report_json = get_model_report_json(repo_id, token)
        if report_json:
            print(f"✓ Report generated for: {report_json.get('repo_id')}")
            print(f"  Architecture: {report_json.get('architecture')}")
            print(f"  Parameters: {report_json.get('parameters')}")
        else:
            print("✗ Report generation returned None")
    except Exception as e:
        print(f"✗ Failed to generate JSON report: {e}")

    # 8️⃣ Test get_model_report_md
    print("\n8. Generating Markdown report...")
    report_md = None
    try:
        report_md = get_model_report_md(repo_id, token)
        if report_md:
            print(f"✓ Markdown report generated ({len(report_md)} chars)")
            print(f"\nPreview:\n{report_md[:200]}...")
        else:
            print("✗ Markdown report returned None")
    except Exception as e:
        print(f"✗ Failed to generate Markdown report: {e}")

    # 9️⃣ Test save_model_report
    print("\n9. Saving report to file...")
    try:
        save_model_report(repo_id, md_path="test_model_report.md", token=token)
        print("✓ Markdown report saved as test_model_report.md")
    except Exception as e:
        print(f"✗ Failed to save report: {e}")

    # 🔟 Test GPU recommendation
    print("\n10. Testing GPU recommendations...")
    gpu_specs = {"name": "RTX 3090", "memory_gb": 24}
    recommended = recommend_models_for_gpu(gpu_specs)
    print(f"✓ Recommended models for {gpu_specs['name']}:")
    for model in recommended:
        print(f"  - {model}")

    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
