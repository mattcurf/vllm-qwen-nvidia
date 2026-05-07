# Qwen3.6-27B NVFP4 vLLM Stack

> **Status: hobby project, provided as-is.** This repository is a personal
> single-GPU configuration that the maintainer happens to find useful. It is
> not a product, has no SLA, and may lag behind upstream vLLM, model, or
> driver changes. Expect to read the configs and adapt them to your own
> hardware. There is no warranty of any kind — see `LICENSE`
> (Apache License 2.0) for the formal terms, and `CONTRIBUTING.md` if you'd
> like to send a fix or improvement.

This directory contains a Dockerized vLLM setup for serving `Qwen3.6-27B` as a fully GPU-accelerated OpenAI-compatible API on port `8888`, tuned for a single NVIDIA RTX 5090 (32 GB) with the multimodal `text + image` checkpoint `sakamakismile/Qwen3.6-27B-NVFP4`.

The defaults:
- current nightly vLLM container via `vllm/vllm-openai:cu129-nightly`
- `Qwen3.6-27B` NVFP4 checkpoint
- `MTP = 2`
- `204,800` token context
- multimodal enabled for text + image
- `flashinfer-cutlass` NVFP4 kernel path because the current `marlin` path crashes on this checkpoint
- OpenAI-compatible server on `http://localhost:8888/v1`

## Important Constraint

The public `~80 t/s @ 218k` Reddit recipe for a single RTX 5090 uses the text-only checkpoint [`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`](https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP), which physically removes the vision tower. That recipe cannot satisfy a `text + image` requirement.

This repo keeps the multimodal vision tower enabled and still configures native MTP at `2`, but the exact `60+ tokens/sec with images at full 204,800-token context` target depends on prompt structure, image size, driver stack, and current upstream Blackwell kernel behavior. The stack here is set up to maximize the chance of hitting that target without dropping image support or using CPU offload.

## Prerequisites

- Linux host with a recent NVIDIA Blackwell-capable driver
- Docker Engine + Compose plugin
- NVIDIA Container Toolkit configured for Docker

Verify GPU passthrough before building:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

## Files

- `Dockerfile` builds a small wrapper image on top of the CUDA 12.9 nightly vLLM image.
- `docker-compose.yml` starts the server with GPU access and the required vLLM flags.
- `.env.example` contains the tuning knobs for a single RTX 5090.
- `scripts/serve.sh` converts env vars into the final `vllm serve` command.
- `scripts/benchmark_multimodal.py` sends one text+image request and prints measured tokens/sec.

## Quick Start

```bash
# Edit .env if you want to change the defaults.
docker compose up -d --build
docker compose logs -f
```

When the model finishes loading, verify the endpoint:

```bash
curl http://localhost:8888/v1/models
```

## Multimodal Benchmark

With a local image:

```bash
python3 scripts/benchmark_multimodal.py \
  --image /path/to/image.jpg \
  --prompt "Describe this image and extract all visible text."
```

With a remote image URL:

```bash
python3 scripts/benchmark_multimodal.py \
  --image-url https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg \
  --prompt "Describe the scene in detail."
```

The script prints:

- elapsed seconds
- completion token count
- total token count
- measured completion tokens per second

## Raw Docker Run Equivalent

If you prefer `docker run` instead of Compose, the equivalent launch looks like this:

```bash
docker build \
  --build-arg VLLM_BASE_IMAGE=vllm/vllm-openai:cu129-nightly \
  -t qwen3.6-vllm-local:latest .

docker run --rm \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8888:8888 \
  -v "$(pwd)/hf-cache:/root/.cache/huggingface" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  -e PORT=8888 \
  -e MODEL_ID=sakamakismile/Qwen3.6-27B-NVFP4 \
  -e SERVED_MODEL_NAME=qwen3.6-27b-nvfp4-multimodal \
  -e MAX_MODEL_LEN=204800 \
  -e GPU_MEMORY_UTILIZATION=0.95 \
  -e KV_CACHE_DTYPE=fp8 \
  -e MTP_TOKENS=3 \
  -e SPECULATIVE_METHOD=mtp \
  -e MAX_NUM_SEQS=1 \
  -e MAX_NUM_BATCHED_TOKENS=2048 \
  -e LIMIT_MM_PER_PROMPT_JSON='{"image":{"count":1,"width":1024,"height":1024},"video":0}' \
  -e MM_PROCESSOR_CACHE_TYPE=shm \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
  -e PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512' \
  -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  -e OMP_NUM_THREADS=1 \
  qwen3.6-vllm-local:latest
```

The required GPU reachability flags are the important part:

- `--runtime nvidia`
- `--gpus all`
- `--ipc=host`

## Default Tuning Choices

- `MODEL_ID=sakamakismile/Qwen3.6-27B-NVFP4`
- `MAX_MODEL_LEN=204800`
- `MTP_TOKENS=3`
- `KV_CACHE_DTYPE=fp8`
- `GPU_MEMORY_UTILIZATION=0.95`
- `MAX_NUM_SEQS=1`
- `MAX_NUM_BATCHED_TOKENS=2048`
- `LIMIT_MM_PER_PROMPT_JSON={"image":{"count":1,"width":1024,"height":1024},"video":0}`
- `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`

Why these defaults:

- `fp8` KV cache is the supported compressed cache format for this Qwen3.6 decoder-attention path in vLLM and is the right starting point for fitting long context on a 32 GB card.
- `MAX_NUM_SEQS=1` and `MAX_NUM_BATCHED_TOKENS=2048` bias the server toward single-request interactive performance and long-context fit.
- `LIMIT_MM_PER_PROMPT_JSON` keeps image support enabled while disabling video profiling overhead.
- `flashinfer-cutlass` is pinned because the current `marlin` kernel path crashes for this checkpoint with `size_n = 96 is not divisible by tile_n_size = 64`.

## If You Need To Adjust

If the multimodal model does not fit cleanly at full `204800` context on your exact driver/runtime combination, edit `.env` in this order:

1. lower `MAX_MODEL_LEN` to `196608`
2. lower `MAX_MODEL_LEN` again to `131072`
3. temporarily unset `VLLM_NVFP4_GEMM_BACKEND` to let vLLM auto-select if a newer nightly fixes backend selection
4. lower `MAX_NUM_BATCHED_TOKENS` to `1024`

If model initialization fails immediately with an out-of-memory error, check for another GPU-heavy container first:

```bash
docker ps --format '{{.Names}}\t{{.Status}}'
docker stop vllm-server
```

On this host, a separate `vllm-server` container was observed holding roughly 30 GB of VRAM, which is enough to make this stack fail even with otherwise-correct settings.

For highest decode throughput when you do not need images, the proven single-5090 path is the separate text-only checkpoint from the Reddit thread. This repo intentionally does not default to that model because you asked for `text + image` support.

## CUDA Image Note

On this RTX 5090 host, `vllm/vllm-openai:cu130-nightly` reproduces a PyTorch CUDA init failure:

`Error 804: forward compatibility was attempted on non supported HW`

`vllm/vllm-openai:cu129-nightly` works correctly on the same machine and still provides a current nightly vLLM build, so this repo now defaults to `cu129-nightly`.

## License

Licensed under the Apache License, Version 2.0 — see `LICENSE` for the
full text. Contributions are accepted under the same license; see
`CONTRIBUTING.md`.

This wrapper does **not** redistribute the upstream vLLM image, the Qwen3.6
model weights, or any quantized checkpoints. Those are pulled at runtime
from their original sources and remain subject to the licenses set by their
respective publishers (vLLM, Alibaba/Qwen, the quant author on Hugging Face,
etc.). Confirm those licenses fit your intended use before deploying.

## Disclaimer

This stack is a personal experiment shared in case it is useful to others.
It is not affiliated with or endorsed by Anthropic, the vLLM project,
Alibaba/Qwen, NVIDIA, or any quant author referenced here. Defaults change
as nightly vLLM and the surrounding ecosystem evolve; always read the
current `.env.example` and the launch logs before assuming a documented
setting still applies.
