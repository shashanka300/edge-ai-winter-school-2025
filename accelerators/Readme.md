
# LLM Inference Workshop – Jetson Xavier Setup

This document explains how to log into the Jetson Xavier device, activate the Python environment, and locate the provided LLM inference scripts used in the workshop.

---

## 1️ SSH into the Jetson Device

Connect to the Jetson as `edgeuser` using the following IP address:

```bash
ssh edgeuser@10.24.24.46
ssh edgeuser@10.24.24.47
ssh edgeuser@10.24.24.48
```

## Navigate to the Workshop Directory

```bash
cd /media/ssd/workshop
```
## Activate the Python Virtual Environment

```bash
source venv/bin/activate
```

## Verify Workshop Files

```bash
ls
```

You should see the following files and folders:

```bash
benchmark_llm.py
run_llm.py
run_llm_batched.py
throughput_vs_batch.png
gpt-neo-1.3B
torch-1.10.0-cp36-cp36m-linux_aarch64.whl
venv
```

## File Overview

- **`run_llm.py`**
  Runs single-prompt LLM inference using manual autoregressive decoding. Designed to work reliably on Jetson devices without distributed or FSDP support.

- **`run_llm_batched.py`**
  Runs batched LLM inference to study throughput as a function of batch size.

- **`benchmark_llm.py`**
  Benchmarks token generation performance, including latency and tokens/sec.

- **`gpt-neo-1.3B/`**
  Local HuggingFace checkpoint for the GPT-Neo-1.3B model. The model is loaded entirely offline from this directory.

- **`venv/`**
  Preconfigured Python virtual environment containing PyTorch and required dependencies.

- **`throughput_vs_batch.png`**
  Pre-generated plot showing throughput scaling with batch size.


## Running a Script

```bash
python3 run_llm.py
```

```bash
python3 run_llm_batched.py
```

```bash
python3 benchmark_llm.py
```
