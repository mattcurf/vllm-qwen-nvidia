#!/usr/bin/env bash

set -euo pipefail

MODEL_ID="${MODEL_ID:-Lorbus/Qwen3.6-27B-int4-AutoRound}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b-int4-autoround}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-190484}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
LIMIT_MM_PER_PROMPT_JSON="${LIMIT_MM_PER_PROMPT_JSON:-}"
MM_PROCESSOR_CACHE_TYPE="${MM_PROCESSOR_CACHE_TYPE:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-3}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MTP_TOKENS="${MTP_TOKENS:-1}"
SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-mtp}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-0}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  unset HF_TOKEN
fi

if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  unset HUGGING_FACE_HUB_TOKEN
fi

if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  unset PYTORCH_CUDA_ALLOC_CONF
fi

if [[ -z "${CUDA_DEVICE_MAX_CONNECTIONS:-}" ]]; then
  unset CUDA_DEVICE_MAX_CONNECTIONS
fi

if [[ -z "${VLLM_USE_FLASHINFER_SAMPLER:-}" ]] || ! [[ "$VLLM_USE_FLASHINFER_SAMPLER" =~ ^[0-9]+$ ]]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi

if [[ -z "${OMP_NUM_THREADS:-}" ]] || ! [[ "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=4
fi

args=(
  serve
  "$MODEL_ID"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --dtype "$DTYPE"
  --reasoning-parser "$REASONING_PARSER"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --max-num-seqs "$MAX_NUM_SEQS"
)

if [[ -n "$LIMIT_MM_PER_PROMPT_JSON" ]]; then
  args+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT_JSON")
fi

if [[ -n "$MM_PROCESSOR_CACHE_TYPE" ]]; then
  args+=(--mm-processor-cache-type "$MM_PROCESSOR_CACHE_TYPE")
fi

if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
  args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  args+=(--enable-prefix-caching)
fi

if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
  args+=(--enable-chunked-prefill)
fi

if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
  args+=(--enable-auto-tool-choice)
fi

if [[ -n "$TOOL_CALL_PARSER" ]]; then
  args+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi

if [[ -n "$CHAT_TEMPLATE" ]]; then
  args+=(--chat-template "$CHAT_TEMPLATE")
fi

if [[ -n "$CUDAGRAPH_MODE" ]]; then
  args+=(--compilation-config.cudagraph_mode "$CUDAGRAPH_MODE")
fi

if [[ "$MTP_TOKENS" =~ ^[0-9]+$ ]] && (( MTP_TOKENS > 0 )); then
  spec_json=$(printf '{"method":"%s","num_speculative_tokens":%s}' "$SPECULATIVE_METHOD" "$MTP_TOKENS")
  args+=(--speculative-config "$spec_json")
fi

if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
  read -r -a extra_args <<<"$EXTRA_VLLM_ARGS"
  args+=("${extra_args[@]}")
fi

printf 'Launching vLLM with model=%s port=%s max_model_len=%s kv_cache_dtype=%s mtp=%s\n' \
  "$MODEL_ID" "$PORT" "$MAX_MODEL_LEN" "$KV_CACHE_DTYPE" "$MTP_TOKENS"

exec vllm "${args[@]}"
