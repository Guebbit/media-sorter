"""Reading image files — the only module that knows about Pillow.

Callers need pixels for different reasons: the scanner wants headers plus a
perceptual hash, the detector wants an array, the vision engine wants a
downscaled JPEG. Sharing the decode path means EXIF orientation, HEIC support
and the decompression-bomb limit are handled once instead of several times.

Videos reach exactly one function here — `to_jpeg_bytes`, which asks `video`
for a frame — because "a picture of this file" is a question worth answering
for an mp4. Everything else is for still images only.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import ExifTags, Image, ImageOps
from pillow_heif import register_heif_opener

from .video import is_video

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

#: dHash grid: one extra column so each row yields 8 left/right comparisons.
_PHASH_WIDTH = 9
_PHASH_HEIGHT = 8


@dataclass(frozen=True, slots=True)
class ImageMeta:
    """What a header read can tell us about a file."""

    width: int | None = None
    height: int | None = None
    format: str | None = None
    taken_at: str | None = None


def inspect(path: str | Path) -> tuple[ImageMeta, str | None]:
    """Header metadata and a perceptual hash, from a single decode.

    What the scanner calls for every new/changed file: dimensions, format,
    EXIF capture date, and a dHash for near-duplicate detection all need the
    same open file, so indexing a photo opens it once here rather than once
    per thing wanted from it. Never raises: an unreadable or corrupt file is
    still a real file worth indexing, it just has no metadata and no phash.
    """
    try:
        with Image.open(path) as img:
            meta = ImageMeta(img.size[0], img.size[1], img.format, _taken_at(img))
            return meta, _dhash(img)
    except Exception as exc:  # noqa: BLE001 - a broken file must not fail a scan
        log.debug("inspect failed for %s: %s", path, exc)
        return ImageMeta(), None


def _dhash(img: Image.Image) -> str:
    """The dHash bits for an already-open image, as 16 hex chars.

    Two images that look the same to a human end up with a small Hamming
    distance between their hashes, even when their bytes differ completely —
    a re-save, a different format, a resize. That is what makes this useful
    for near-duplicate detection where `filesystem.sha256_file` (exact bytes)
    is blind.
    """
    grayscale = ImageOps.exif_transpose(img).convert("L").resize(
        (_PHASH_WIDTH, _PHASH_HEIGHT), Image.LANCZOS
    )
    # `tobytes` rather than `getdata`: mode "L" is one byte per pixel in row
    # order, so the buffer *is* the pixel list, and `getdata` is deprecated
    # (removed in Pillow 14) in favour of an API that does not exist in the
    # oldest Pillow this project supports.
    pixels = grayscale.tobytes()
    bits = 0
    for row in range(_PHASH_HEIGHT):
        offset = row * _PHASH_WIDTH
        for col in range(_PHASH_WIDTH - 1):
            bits = (bits << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return f"{bits:016x}"


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


def _preview(path: str | Path) -> Image.Image:
    """One picture standing for whatever is at `path`.

    For a video that is a frame from halfway in, not an error: every caller of
    `to_jpeg_bytes` wants "show me this file", and Pillow cannot open an mp4.
    The import is local because it pulls in OpenCV, which a run with no videos
    in it should never load.
    """
    if is_video(path):
        from . import video

        return Image.fromarray(video.to_rgb(video.middle_frame(path)))
    return load_rgb(path)


def to_jpeg_bytes(path: str | Path, max_edge: int, quality: int = 88) -> bytes:
    """A downscaled JPEG's raw bytes — what the web UI's thumbnail route serves,
    and what `to_jpeg_base64` encodes for an inference API. Handles videos, by
    way of `_preview`."""
    img = _preview(path)
    longest = max(img.size)
    if longest > max_edge:
        scale = max_edge / longest
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS
        )
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def to_jpeg_base64(path: str | Path, max_edge: int, quality: int = 88) -> str:
    """A downscaled JPEG, base64-encoded — what an HTTP inference API wants."""
    return base64.b64encode(to_jpeg_bytes(path, max_edge, quality)).decode("ascii")
