import torch
import time
import sys
import argparse
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Run Batched LLM Inference on Jetson")
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    choices=[1, 2, 4, 8, 16, 32],
    help="Number of inputs to process in parallel (Default: 4)"
)
args = parser.parse_args()

# --- CONFIGURATION ---
MODEL_NAME = "/media/ssd/workshop/gpt-neo-1.3B"
NEW_TOKENS = 150  # Generate exactly this many tokens
BATCH_SIZE = args.batch_size

# Base prompts to cycle through
BASE_PROMPTS = [
    "The future of artificial intelligence depends on",
    "when will AGI come?",
    "The biggest challenge for autonomous robots is",
    "To optimize neural networks on edge devices, one must",
    "The history of computing began when",
    "In the context of machine learning, a transformer is",
    "NVIDIA Jetson devices are designed to",
    "Python is the most popular language for AI because"
]


# Generate the exact list of prompts for the requested batch size
# We repeat the base list until we have enough, then slice it.
repeats = math.ceil(BATCH_SIZE / len(BASE_PROMPTS))
BATCH_PROMPTS = (BASE_PROMPTS * repeats)[:BATCH_SIZE]
# ---------------------

print(f"--- SYSTEM CHECK (Batch Size: {BATCH_SIZE}) ---")
if not torch.cuda.is_available():
    print("CRITICAL ERROR: CUDA (GPU) is not available!")
    sys.exit(1)

device = torch.device("cuda:0")
print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")

print(f"\n--- Loading Model from: {MODEL_NAME} ---")

# 1. Load Tokenizer & Configure Padding
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # Critical for generation

# 2. Load Model
# Using FP16 to ensure larger batches fit in VRAM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    local_files_only=True
).to(device)

model.config.pad_token_id = model.config.eos_token_id

print(f"✅ Model Loaded on {model.device}")
print(f"--- Tokenizing {len(BATCH_PROMPTS)} Prompts ---")

# 3. Tokenize Batch
inputs = tokenizer(BATCH_PROMPTS, return_tensors="pt", padding=True).to(device)
input_tokens = inputs['input_ids'].shape[1]

print(f"Input Shape: {inputs['input_ids'].shape} (Batch x SeqLen)")
print("--- Generating... (Warmup not included, this is a raw run) ---")

start_time = time.time()

# 4. Generate
with torch.no_grad():
    generated_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        min_new_tokens=NEW_TOKENS,
        max_new_tokens=NEW_TOKENS,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

end_time = time.time()
duration = end_time - start_time
total_tokens_gen = BATCH_SIZE * NEW_TOKENS
tokens_per_sec = total_tokens_gen / duration

print(f"\n✅ Batch Complete in {duration:.2f}s")
print(f"📊 Total Tokens Generated: {total_tokens_gen}")
print(f"⚡ Throughput: {tokens_per_sec:.2f} tokens/sec")
print("="*60)

# 5. Decode & Print (Only print first 2 to keep output clean if batch is large)
print("--- Sample Outputs (First 2) ---")
decoded_outputs = tokenizer.batch_decode(generated_ids[:2], skip_special_tokens=True)

for i, output in enumerate(decoded_outputs):
    print(f"\n[PROMPT {i+1}]: {BATCH_PROMPTS[i]}")
    print("-" * 30)
    print(output)
    print("="*60)
