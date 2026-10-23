"""Pages and pictures as model input: render a PDF page, encode an image for an OpenAI-compatible API."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pymupdf
from PIL import Image

# 200 DPI keeps small digits readable for a VLM and is the resolution of our scans.
VLM_DPI = 200


def render_page(pdf: str | Path, page: int, dpi: int = VLM_DPI) -> Image.Image:
    """Page `page` (1-based) of a PDF as a greyscale picture."""
    with pymupdf.open(pdf) as doc:
        pix = doc[page - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB" if img.mode not in ("L", "RGB") else img.mode).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def image_part(image: Image.Image | bytes | str | Path) -> dict:
    """One image as a message part: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    data = image if isinstance(image, bytes) else jpeg_bytes(image)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}}


def vision_message(text: str, *images) -> dict:
    """A user message with text and one or more images."""
    return {"role": "user", "content": [{"type": "text", "text": text}, *[image_part(i) for i in images]]}


def draw_layout(page_img: Image.Image, layout: list[dict], dpi: int) -> Image.Image:
    """Boxes of a document VLM (in PDF points) drawn on a page picture rendered at `dpi`."""
    from PIL import ImageDraw, ImageFont

    img = page_img.convert("RGB")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=15)
    k = dpi / 72
    colors = {"Table": (220, 90, 0), "Section-header": (0, 90, 200), "Caption": (130, 0, 160),
              "Page-header": (120, 120, 120), "Page-footer": (120, 120, 120), "Picture": (0, 150, 60)}
    for block in layout:
        x0, y0, x1, y1 = (v * k for v in block["bbox_pt"])
        c = colors.get(block["category"], (200, 0, 0))
        d.rectangle([x0, y0, x1, y1], outline=c, width=3)
        d.text((x1 - 4, y0 + 2), block["category"], fill=c, font=font, anchor="ra")
    return img
