"""Check that a model profile answers text and reads an image. Usage:

    uv run python -m plant_assistant.smoke_test            # profile "local"
    uv run python -m plant_assistant.smoke_test course     # course Qwen
"""

from __future__ import annotations

import sys

from PIL import Image, ImageDraw, ImageFont

from .config import get_profile, load_env
from .images import vision_message
from .llm import LLMClient


def test_image() -> Image.Image:
    img = Image.new("L", (640, 200), 255)
    d = ImageDraw.Draw(img)
    try:
        import matplotlib
        from pathlib import Path

        font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"), 56)
    except Exception:
        font = ImageFont.load_default()
    d.text((30, 60), "PUMP P-417", fill=0, font=font)
    return img


def check(profile_name: str) -> bool:
    env_path = load_env()
    profile = get_profile(profile_name)
    print(f"profile : {profile.name} ({profile.description})")
    print(f"model   : {profile.model}")
    print(f"url     : {profile.base_url or '<missing>'}")
    print(f".env    : {env_path or '<not found>'}")
    missing = profile.missing_settings()
    if missing:
        print("FAIL    : " + "; ".join(missing))
        return False
    client = LLMClient(profile, max_network_retries=0)
    reply = client.chat([{"role": "user", "content": "Reply with one word: OK"}])
    if not reply.ok:
        print(f"FAIL    : {reply.error}")
        if "localhost" in profile.base_url:
            print("hint    : is Ollama running? Is the model pulled? `ollama pull qwen3.5:4b-q4_K_M`")
        return False
    print(f"text    : {reply.content.strip()[:40]!r} in {reply.latency_s:.1f}s (the first call can be slow: model loading)")
    seen = client.chat([vision_message("What is written in the image? Answer with the text only.", test_image())])
    if not seen.ok or "417" not in seen.content:
        print(f"FAIL    : the model does not read images: {seen.error or seen.content[:120]!r}")
        print("hint    : the model must be a VLM (vision). For Ollama: `ollama show <model>` lists 'vision'.")
        return False
    print(f"image   : {seen.content.strip()[:40]!r} in {seen.latency_s:.1f}s, {seen.prompt_tokens} prompt tokens")
    print("OK")
    return True


if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1] if len(sys.argv) > 1 else "local") else 1)
