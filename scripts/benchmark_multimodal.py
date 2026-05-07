#!/usr/bin/env python3

import argparse
import base64
import json
import mimetypes
import pathlib
import time
import urllib.error
import urllib.request


def image_payload(image_path: str | None, image_url: str | None) -> dict:
    if image_url:
        return {"type": "image_url", "image_url": {"url": image_url}}

    if not image_path:
        raise ValueError("either --image or --image-url is required")

    path = pathlib.Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "image/jpeg"

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def render_message_text(message: dict) -> str:
    sections: list[str] = []

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        sections.append(f"[reasoning]\n{reasoning}")

    content = message.get("content")
    if isinstance(content, str) and content:
        sections.append(f"[content]\n{content}")
    elif isinstance(content, list):
        text_chunks: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_text = item.get("text")
            if item.get("type") in {"text", "output_text"} and isinstance(item_text, str) and item_text:
                text_chunks.append(item_text)
        if text_chunks:
            sections.append(f"[content]\n{'\n'.join(text_chunks)}")

    return "\n\n".join(sections) if sections else "(no assistant text returned)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one text+image request to a vLLM OpenAI-compatible endpoint and print tokens/sec.")
    parser.add_argument("--url", default="http://localhost:8888/v1/chat/completions")
    parser.add_argument("--model", default="qwen3.6-27b-nvfp4-multimodal")
    parser.add_argument("--image")
    parser.add_argument("--image-url")
    parser.add_argument("--prompt", default="Describe this image in detail and extract any visible text.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    image_payload(args.image, args.image_url),
                ],
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8"))
        return 1

    elapsed = time.perf_counter() - started
    usage = body.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    content = render_message_text(body["choices"][0]["message"])
    tokens_per_second = (completion_tokens / elapsed) if elapsed and completion_tokens else 0.0

    print(f"elapsed_s={elapsed:.2f}")
    print(f"completion_tokens={completion_tokens}")
    print(f"total_tokens={total_tokens}")
    print(f"tokens_per_second={tokens_per_second:.2f}")
    print()
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
