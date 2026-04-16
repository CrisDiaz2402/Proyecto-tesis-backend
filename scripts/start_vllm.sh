#!/bin/bash
# Lanzar vLLM con modelo cuantizado AWQ para GTX 3050 (4 GB VRAM)
# Ajustado para entornos donde el sistema operativo también consume VRAM

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct-AWQ \
    --quantization awq \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.75 \
    --max-num-seqs 4 \
    --port 8001 \
    --host 0.0.0.0 \
    --served-model-name llama3-local