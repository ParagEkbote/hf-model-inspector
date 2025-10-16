from hf_model_inspector import get_lora_info

repo_id = "luminolous/llama-wikindo-lora"

info = get_lora_info(repo_id)

if info:
    print("✅ LoRA Configuration Summary:")
    for k, v in info.items():
        print(f"{k}: {v}")
else:
    print("❌ No LoRA configuration found for this model.")
