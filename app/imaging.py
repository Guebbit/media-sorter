"""Reading image files — the only module that knows about Pillow.

Three callers need pixels for three different reasons: the scanner wants
headers, the detector wants an array, the vision engine wants a downscaled
JPEG. Sharing the decode path means EXIF orientation, HEIC support and the
decompression-bomb limit are handled once instead of three times.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import ExifTags, Image, ImageOps
from pillow_heif import register_heif_opener

log = logging.getLogger(__name__)

# Huge panoramas are legitimate here; we only ever read headers or downscale.
Image.MAX_IMAGE_PIXELS = None

# iPhone libraries are mostly HEIC, and .heic/.heif are in the default extension
# list, so the opener is not optional: registering it here means every caller of
# this module gets it for free.
register_heif_opener()

_DATETIME_TAGS = {
    tag_id
    for tag_id, name in ExifTags.TAGS.items()
    if name in {"DateTimeOriginal", "DateTimeDigitized", "DateTime"}
}


@dataclass(frozen=True, slots=True)
class ImageMeta:
    """What a header read can tell us about a file."""

    width: int | None = None
    height: int | None = None
    format: str | None = None
    taken_at: str | None = None


def probe(path: str | Path) -> ImageMeta:
    """Dimensions, format and EXIF capture date, without decoding the pixels.

    Never raises: an unreadable or corrupt file is still a real file worth
    indexing, it just has no metadata.
    """
    try:
        with Image.open(path) as img:
            return ImageMeta(img.size[0], img.size[1], img.format, _taken_at(img))
    except Exception as exc:  # noqa: BLE001 - a broken file must not fail a scan
        log.debug("probe failed for %s: %s", path, exc)
        return ImageMeta()


def _taken_at(img: Image.Image) -> str | None:
    """The EXIF capture timestamp, whichever of `_DATETIME_TAGS` is present
    first — or None if there's no EXIF, or none of those tags are set."""
    try:
        exif = img.getexif()
    except Exception:  # noqa: BLE001 - corrupt EXIF is common and harmless
        return None
    for tag_id in _DATETIME_TAGS:
        value = exif.get(tag_id)
        if value:
            return str(value).strip()
    return None


def load_rgb(path: str | Path) -> Image.Image:
    """Decode to RGB with EXIF orientation applied. Raises on a broken file."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def to_jpeg_base64(path: str | Path, max_edge: int, quality: int = 88) -> str:
    """A downscaled JPEG, base64-encoded — what an HTTP inference API wants."""
    img = load_rgb(path)
    longest = max(img.size)
    if longest > max_edge:
        scale = max_edge / longest
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS
        )
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
