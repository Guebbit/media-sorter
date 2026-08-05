"""YOLO detection.

One model instance, batched inference. Images are decoded on a thread pool
while the GPU works on the previous batch, which is where the throughput comes
from. Nothing here knows about categories, rules or the semantic pass — it turns
paths into boxes.

A video is the same work in a different shape: `detect_video` samples stills
across the file and runs them as one batch, then hands the frames' boxes to
`merge_frames` for the single answer the index has room for.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .. import imaging, video
from ..config import DetectSettings
from ..domain.detection import Detection, merge_frames

log = logging.getLogger(__name__)

# Must be set before ultralytics is imported anywhere.
os.environ.setdefault("YOLO_VERBOSE", "false")


def resolve_model(name: str, search_paths: Sequence[Path]) -> str:
    """Find the weights locally, or fetch them once into the first search path.

    A deployment that ships weights alongside the code adds its directory to
    `DetectSettings.model_search_paths`; the code itself has no idea where that
    might be.
    """
    candidate = Path(name)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    for directory in (*search_paths, Path.cwd()):
        local = Path(directory) / candidate.name
        if local.exists():
            return str(local)

    if not search_paths:
        return name
    cache = Path(search_paths[0])
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / candidate.name
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(str(target))
        if target.exists():
            return str(target)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not pre-download %s: %s", name, exc)
    return name  # let ultralytics resolve it (needs network)


def _resolve_class_ids(model_name: str, names: dict[int, str], classes: Iterable[str]) -> list[int]:
    """The model's internal ids for the requested classes.

    Separated from `Detector.__init__` so "do the rules and this model's
    vocabulary agree" is one focused check — model loading and rule validation
    are two different reasons this could fail, and each gets its own error.
    Raises `ValueError` if nothing was requested, or if the model's weights
    cannot produce a class the rules ask for.
    """
    wanted = {c.lower() for c in classes}
    if not wanted:
        raise ValueError("no classes to detect — the rules define them")
    class_ids = [i for i, n in names.items() if n.lower() in wanted]
    missing = wanted - {names[i].lower() for i in class_ids}
    if missing:
        raise ValueError(
            f"model {model_name} cannot detect {sorted(missing)}; "
            f"available classes: {sorted(names.values())}"
        )
    return class_ids


class Detector:
    """Wraps one loaded model. Built per stage run, closed when the stage ends."""

    def __init__(self, settings: DetectSettings, classes: Iterable[str], conf_floor: float,
                decode_workers: int = 4):
        """Load one YOLO model and prepare it to detect only `classes`.

        `conf_floor` is the lowest `review_confidence` any class in the active
        ruleset asks for — the most permissive floor, so nothing any rule's
        band needs gets filtered out by YOLO before it ever reaches the
        decision layer. Bands are per class, so the caller
        (`services.processing`) computes it from `RuleSet.class_bands()`.

        This is the one place `ultralytics.YOLO(...)` is instantiated: it reads
        the weights file at `self.model_path` into (typically) GPU memory, which
        is why a `Detector` is built once per stage run and reused for every
        batch, not once per image. The import is inside the method, not at
        module level, because `ultralytics` drags in `torch` — a multi-second,
        multi-hundred-MB cost that every command which never touches the
        detector (`config show`, `rules show`, ...) would otherwise pay too.
        """
        from concurrent.futures import ThreadPoolExecutor
        from ultralytics import YOLO  # lazy: heavy, and most commands never need it

        self.settings = settings
        self.conf_floor = conf_floor
        self.model_path = resolve_model(settings.model, settings.model_search_paths)
        log.info("loading YOLO weights: %s", self.model_path)
        self.model = YOLO(self.model_path)
        self.model_name = Path(self.model_path).name

        # `self.model.names` is YOLO's own {internal id: class name} table —
        # fixed by the weights file, not something we choose. `class_ids` is
        # the subset of those ids we will actually ask predict() for, computed
        # once here so every later `detect_batch` call reuses it.
        names: dict[int, str] = self.model.names
        self.class_ids = _resolve_class_ids(self.model_name, names, classes)
        self.names = names
        # Decoding (disk + Pillow) happens off the thread that calls predict(),
        # so the GPU is never left idle waiting for the next batch's files to load.
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, decode_workers), thread_name_prefix="decode"
        )

    def close(self) -> None:
        """Stop the decode thread pool. Does not touch the model itself —
        there is nothing to release beyond letting Python garbage-collect it."""
        self._pool.shutdown(wait=False)

    def _load(self, path: str) -> np.ndarray | None:
        """One image, decoded to a BGR array ready for `model.predict`.
        None on any decode failure, so one corrupt file only drops itself from
        the batch instead of raising out of the thread pool.

        BGR, not the RGB Pillow hands back: ultralytics reads a numpy array the
        way OpenCV wrote it (`LoadPilAndNumpy._single_check` — "NumPy color
        inputs are assumed to use OpenCV-compatible BGR order"), so passing RGB
        silently swaps red and blue on every photo and costs accuracy for free.
        Video frames arrive from `video` in that order already.
        """
        try:
            return np.ascontiguousarray(np.asarray(imaging.load_rgb(path))[:, :, ::-1])
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop a batch
            log.warning("cannot decode %s: %s", path, exc)
            return None

    def detect_batch(self, paths: list[str]) -> dict[str, list[Detection] | None]:
        """Run YOLO inference on every image in `paths`, once, as one batch.

        Still images only — the stage sends videos to `detect_video` instead.
        None means the image could not be read; the caller records an error.
        Decoding happens on the thread pool first (see `_load`) so `_predict`
        is handed ready-made numpy arrays.
        """
        images = list(self._pool.map(self._load, paths))
        usable = [(path, array) for path, array in zip(paths, images) if array is not None]
        results: dict[str, list[Detection] | None] = {
            path: None for path, array in zip(paths, images) if array is None
        }
        if not usable:
            return results

        for (path, _), detections in zip(usable, self._predict([a for _, a in usable])):
            results[path] = detections
        return results

    def detect_video(self, path: str) -> list[Detection] | None:
        """What is in one video, from `settings.video_frames` stills of it.

        The frames are a batch like any other — one predict call for the whole
        file — so a video costs what that many photos cost, and `merge_frames`
        collapses the per-frame boxes into the one answer the index stores per
        file. Sampling and merging both live behind this method so the stage
        above never learns that a video is more than one inference.

        None means no frame could be read at all: a container OpenCV will not
        open, a truncated download. Not an error — the file is still a video,
        and the caller falls back to saying only that.
        """
        try:
            frames = video.sample_frames(path, self.settings.video_frames)
        except Exception as exc:  # noqa: BLE001 - one bad video must not stop a batch
            log.warning("cannot sample %s: %s", path, exc)
            return None
        return merge_frames(self._predict(frames))

    def _predict(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """One forward pass over `images` (BGR arrays), as a single batch.

        The only point this class talks to the GPU (or CPU), for photos and for
        video frames alike — batching is the whole reason this is one call and
        not one per image.
        """
        predictions = self.model.predict(
            images,
            # Keep borderline hits: the review flag is computed from them.
            conf=self.conf_floor,
            classes=self.class_ids,
            imgsz=self.settings.imgsz,
            device=self.settings.device,
            verbose=False,
            stream=False,
        )
        return [self._to_detections(prediction) for prediction in predictions]

    def _to_detections(self, prediction) -> list[Detection]:
        """One YOLO `Results` object (ultralytics' own prediction type) turned
        into our own `Detection` values — the only place ultralytics' box
        format (`.xyxy`, `.cls`, `.conf` tensors) is read at all, so nothing
        downstream of this class needs to know it exists."""
        detections: list[Detection] = []
        boxes = getattr(prediction, "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    cls=self.names[int(box.cls.item())].lower(),
                    confidence=float(box.conf.item()),
                    x1=round(xyxy[0], 2), y1=round(xyxy[1], 2),
                    x2=round(xyxy[2], 2), y2=round(xyxy[3], 2),
                )
            )
        return detections


def available_classes(settings: DetectSettings) -> list[str]:
    """Every class the configured weights can detect — the menu the rules
    editor offers. Loading the model is the only honest source for this: it
    briefly instantiates a second `YOLO` just to read `.names` off it, then
    lets it be garbage-collected — cheaper than keeping a `Detector` around
    for a call the rules editor makes once, not per image."""
    from ultralytics import YOLO

    model = YOLO(resolve_model(settings.model, settings.model_search_paths))
    return sorted({name.lower() for name in model.names.values()})


def describe_device(settings: DetectSettings) -> str:
    """Human-readable answer to "what will `detect` actually run on?" — for
    `mediasort doctor` and the detect command's banner. Reads `torch.cuda`
    directly rather than asking YOLO, since this needs an answer even when no
    `Detector` (and so no loaded model) exists yet."""
    try:
        import torch

        if settings.device and settings.device.startswith("cpu"):
            return "cpu (forced)"
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
        return "cpu (no CUDA visible)"
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc})"
