ARG VLLM_BASE_IMAGE=vllm/vllm-openai:cu129-nightly
FROM ${VLLM_BASE_IMAGE}

ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/serve.sh /opt/vllm/serve.sh

RUN chmod +x /opt/vllm/serve.sh

ENTRYPOINT ["/opt/vllm/serve.sh"]
