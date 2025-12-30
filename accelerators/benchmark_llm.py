import torch
import time
import sys
import re
import threading
import subprocess
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
# --- FIX: Disable tokenizer parallelism to stop fork warnings ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- CONFIGURATION ---
MODEL_PATH = "/media/ssd/workshop/gpt-neo-1.3B"
# You can adjust these batch sizes based on available VRAM
BATCH_SIZES = [1, 2, 4, 8, 16]
NEW_TOKENS = 100
OUTPUT_PLOT = "/media/ssd/workshop/throughput_vs_batch.png"

# Since you confirmed sudo is not needed:
TEGRASTATS_CMD = ["tegrastats", "--interval", "100"]

# Base prompt to duplicate
BASE_PROMPT = "The future of artificial intelligence depends on"

class GPU_Logger:
    def __init__(self):
        self.gpu_usage = []
        self.running = False
        self.process = None
        self.thread = None

    def start(self):
        self.running = True
        self.gpu_usage = []
        try:
            # Launch tegrastats directly
            self.process = subprocess.Popen(
                TEGRASTATS_CMD,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.thread = threading.Thread(target=self._monitor)
            self.thread.start()
        except Exception as e:
            print(f"⚠️ Failed to start tegrastats: {e}")
            self.process = None

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
            if self.thread:
                self.thread.join()

    def _monitor(self):
        # Regex to handle both "GR3D 50%" and "GR3D_FREQ 50%" formats
        pattern = re.compile(r"GR3D(?:_FREQ)?\s+(\d+)%")

        while self.running and self.process:
            line = self.process.stdout.readline()
            if not line:
                break

            # Extract GPU Load
            match = pattern.search(line)
            if match:
                load = int(match.group(1))
                self.gpu_usage.append(load)

def run_benchmark():
    # 1. Setup Device & Model
    print("--- 🚀 INITIALIZING WORKSHOP BENCHMARK ---")
    if not torch.cuda.is_available():
        print("❌ CRITICAL: No GPU found.")
        sys.exit(1)

    device = torch.device("cuda:0")
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"📂 Loading Model from: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Critical for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)
    model.config.pad_token_id = model.config.eos_token_id

    # 2. Storage for Results
    results = {
        "batch_size": [],
        "throughput": [],
        "avg_gpu": []
    }

    # 3. Warmup (Critical for accurate timing)
    print("\n🔥 Warming up GPU (running one batch)...")
    warmup_input = tokenizer([BASE_PROMPT], return_tensors="pt").to(device)
    model.generate(**warmup_input, max_new_tokens=10, min_new_tokens=10)
    print("✅ Warmup complete. Starting measurements.\n")

    # 4. Main Benchmark Loop
    logger = GPU_Logger()

    for b_size in BATCH_SIZES:
        print(f"--- 🧪 Testing Batch Size: {b_size} ---")

        # Prepare inputs
        prompts = [BASE_PROMPT] * b_size
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        input_tokens_count = inputs.input_ids.shape[1]

        # Start Logging
        logger.start()

        # Run Inference
        torch.cuda.synchronize() # Wait for GPU to be ready
        start_time = time.time()

        with torch.no_grad():
            _ = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                min_new_tokens=NEW_TOKENS,
                max_new_tokens=NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id
            )

        torch.cuda.synchronize() # Wait for GPU to finish
        end_time = time.time()

        # Stop Logging
        logger.stop()

        # Calculate Metrics
        total_time = end_time - start_time
        # Throughput = (Input + Output) * Batch_Size / Time
        total_tokens_processed = (input_tokens_count + NEW_TOKENS) * b_size
        throughput = total_tokens_processed / total_time

        # Calculate Avg GPU Usage
        if logger.gpu_usage:
            avg_gpu = sum(logger.gpu_usage) / len(logger.gpu_usage)
        else:
            avg_gpu = 0

        print(f"   ⏱️  Time Taken:   {total_time:.2f} s")
        print(f"   ⚡ Throughput:   {throughput:.2f} tokens/s")
        print(f"   🔋 Avg GPU Load: {avg_gpu:.1f}%")

        results["batch_size"].append(b_size)
        results["throughput"].append(throughput)
        results["avg_gpu"].append(avg_gpu)

    return results

def plot_results(results):
    print(f"\n🎨 Generating plot...")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Throughput (Blue Line)
    color = 'tab:blue'
    ax1.set_xlabel('Batch Size (Log Scale)', fontsize=12)
    ax1.set_ylabel('Throughput (tokens/sec)', color=color, fontsize=12)
    ax1.plot(results["batch_size"], results["throughput"], color=color, marker='o', linewidth=3, label='Throughput')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log') # Log scale helps visualize 1 vs 16 better
    ax1.set_xticks(results["batch_size"])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # Show "1, 2, 4" instead of "10^0"
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # Plot GPU Usage (Red Dashed Line)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Avg GPU Usage (%)', color=color, fontsize=12)
    ax2.plot(results["batch_size"], results["avg_gpu"], color=color, marker='s', linestyle='--', linewidth=2, label='GPU Usage')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 105)

    # Title and Layout
    plt.title(f'LLM Inference Scaling on Jetson Xavier AGX\n(GPT-Neo 1.3B, FP16)', fontsize=14)
    fig.tight_layout()

    plt.savefig(OUTPUT_PLOT)
    print(f"💾 Plot saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    try:
        data = run_benchmark()
        plot_results(data)
        print("\n✅ Benchmark Completed Successfully.")
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user.")
