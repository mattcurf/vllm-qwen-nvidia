# Qwen3.6-27B Int4 AutoRound vLLM Stack

> **Status: hobby project, provided as-is.** This repository is a personal
> single-GPU configuration that the maintainer happens to find useful. It is
> not a product, has no SLA, and may lag behind upstream vLLM, model, or
> driver changes. Expect to read the configs and adapt them to your own
> hardware. There is no warranty of any kind — see `LICENSE`
> (Apache License 2.0) for the formal terms, and `CONTRIBUTING.md` if you'd
> like to send a fix or improvement.

This directory contains a Dockerized vLLM setup for serving `Lorbus/Qwen3.6-27B-int4-AutoRound` as a fully GPU-accelerated OpenAI-compatible API on port `8888`, exposed as `qwen3.6-27b-int4-autoround` and tuned for a single NVIDIA RTX 5090 (32 GB).

The defaults:
- current nightly vLLM container via `vllm/vllm-openai:cu129-nightly`
- `Lorbus/Qwen3.6-27B-int4-AutoRound`
- served model name `qwen3.6-27b-int4-autoround`
- `MTP = 1`
- `190,484` token context
- `fp8` KV cache
- `GPU_MEMORY_UTILIZATION=0.85`
- automatic tool choice enabled
- default tool-call parser `qwen3_coder`
- multimodal `text + image` support retained from the base Qwen3.6-27B checkpoint
- OpenAI-compatible server on `http://localhost:8888/v1`

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
- `scripts/serve.sh` defines the built-in model and tuning defaults, then converts env vars into the final `vllm serve` command.
- `scripts/benchmark_multimodal.py` sends one text+image request and prints measured tokens/sec.

## Persistent Caches

Compose preserves the two cache directories that materially affect startup
time:

- `${HF_CACHE:-./hf-cache}` mounted at `/root/.cache/huggingface` for model,
  tokenizer, and processor downloads.
- `${VLLM_CACHE:-./vllm-cache}` mounted at `/root/.cache/vllm` for vLLM
  torch.compile/AOT artifacts such as
  `/root/.cache/vllm/torch_compile_cache/...`.

The vLLM cache mount is based on startup logs like:

```text
Using cache directory: /root/.cache/vllm/torch_compile_cache/... for vLLM's torch.compile
saved AOT compiled function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/...
```

## Quick Start

```bash
# Set overrides in your environment or a local .env file if you do not want the built-in defaults.
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
  --model qwen3.6-27b-int4-autoround \
  --image /path/to/image.jpg \
  --prompt "Describe this image and extract all visible text."
```

With a remote image URL:

```bash
python3 scripts/benchmark_multimodal.py \
  --model qwen3.6-27b-int4-autoround \
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
  -v "$(pwd)/vllm-cache:/root/.cache/vllm" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  -e PORT=8888 \
  -e MODEL_ID=Lorbus/Qwen3.6-27B-int4-AutoRound \
  -e SERVED_MODEL_NAME=qwen3.6-27b-int4-autoround \
  -e MAX_MODEL_LEN=190484 \
  -e GPU_MEMORY_UTILIZATION=0.85 \
  -e KV_CACHE_DTYPE=fp8 \
  -e MTP_TOKENS=1 \
  -e SPECULATIVE_METHOD=mtp \
  -e MAX_NUM_SEQS=3 \
  -e TOOL_CALL_PARSER=qwen3_coder \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  qwen3.6-vllm-local:latest
```

The required GPU reachability flags are the important part:

- `--runtime nvidia`
- `--gpus all`
- `--ipc=host`

## Default Tuning Choices

- `MODEL_ID=Lorbus/Qwen3.6-27B-int4-AutoRound`
- `SERVED_MODEL_NAME=qwen3.6-27b-int4-autoround`
- `MAX_MODEL_LEN=190484`
- `MTP_TOKENS=1`
- `KV_CACHE_DTYPE=fp8`
- `GPU_MEMORY_UTILIZATION=0.85`
- `MAX_NUM_SEQS=3`
- `ENABLE_AUTO_TOOL_CHOICE=1`
- `TOOL_CALL_PARSER=qwen3_coder`

Why these defaults:

- `fp8` KV cache is the supported compressed cache format for this Qwen3.6 path in mainline vLLM and is the right starting point for fitting long context on a 32 GB card.
- `MTP_TOKENS=1` keeps native speculative decoding enabled without pushing aggressive draft settings.
- `MAX_NUM_SEQS=3` is a moderate concurrency target for a single-GPU setup; lower it if you want more headroom for larger prompts or stricter memory margins.
- `MAX_MODEL_LEN=190484` keeps a long context while leaving more memory
  headroom than the full Qwen3.6 context target.
- `ENABLE_AUTO_TOOL_CHOICE=1` allows clients such as OpenCode to send
  `tool_choice: "auto"`.
- `TOOL_CALL_PARSER=qwen3_coder` is the working Qwen parser path for
  OpenCode tool calls. The `qwen3_xml` parser has been observed to fail with
  `Expected 'function.name' to be a string` for this setup.

## Tool Calling

Automatic tool calling for clients such as OpenCode is enabled by default:

```bash
ENABLE_AUTO_TOOL_CHOICE=1
TOOL_CALL_PARSER=qwen3_coder
```

`qwen3_coder` is now the built-in parser default. If your deployment has the
enhanced Qwen chat template available, you can also pass it directly instead
of using `EXTRA_VLLM_ARGS`:

```bash
CHAT_TEMPLATE=/models/qwen3.5-enhanced.jinja
```

## If You Need To Adjust

If the model does not fit cleanly at `190484` context on your exact driver/runtime combination, edit `.env` or your exported environment in this order:

1. lower `MAX_MODEL_LEN` to `131072`
2. lower `MAX_NUM_SEQS` to `1`
3. set `MTP_TOKENS=0` if you want the simplest non-speculative path while debugging startup or stability

If model initialization fails immediately with an out-of-memory error, check for another GPU-heavy container first:

```bash
docker ps --format '{{.Names}}\t{{.Status}}'
docker stop vllm-server
```

On this host, a separate `vllm-server` container was observed holding roughly 30 GB of VRAM, which is enough to make this stack fail even with otherwise-correct settings.

Text-only checkpoints can still be faster on a single RTX 5090. This repo intentionally defaults to the AutoRound multimodal checkpoint instead, so `text + image` requests continue to work.

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
current `scripts/serve.sh`, your local `.env` or exported environment, and the launch logs before assuming a documented
setting still applies.
