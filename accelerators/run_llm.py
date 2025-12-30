import torch
import time
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_NAME = "/media/ssd/workshop/gpt-neo-1.3B"
PROMPT = "how to make cookies?"
# ---------------------

print("--- SYSTEM CHECK ---")
# 1. HARD CHECK for GPU
if not torch.cuda.is_available():
    print("CRITICAL ERROR: CUDA (GPU) is not available!")
    print("The script will now exit.")
    sys.exit(1)

device = torch.device("cuda:0")
print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"\n--- Loading Model from: {MODEL_NAME} ---")

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

# 3. Load Model (Explicitly map to GPU during load)
# We use .to(device) immediately to ensure it lands on VRAM
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)
except Exception as e:
    print(f"Error loading model to GPU: {e}")
    sys.exit(1)

print(f"✅ Model is on device: {model.device}")
print("--- Generating text... ---")

# 4. Encode Inputs and FORCE move to GPU
input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(device)

# 5. Generate
start_time = time.time()
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        max_length=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
end_time = time.time()

# 6. Decode & Print
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("\n" + "="*40)
print(f"INPUT PROMPT: {PROMPT}")
print("-" * 40)
print(f"GENERATED OUTPUT ({end_time - start_time:.2f}s):")
print(generated_text)
print("="*40 + "\n")
