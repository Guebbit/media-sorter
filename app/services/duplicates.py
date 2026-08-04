"""Use case: find and review near-duplicate photos.

Where `insights.duplicates` catches byte-identical files, this catches ones
that only look identical — a re-save, a resize, a different format — by
clustering perceptual hashes (`domain.similarity.cluster`).

Its own tool, over its own folders and its own index (`ctx.dupes`), sharing
only the hashing and the schema with the sorting pipeline. Pointing it at a
folder neither indexes that folder for sorting nor runs a model over it.

Marking and dismissing record decisions and touch nothing. `discard_marked` is
the single exception and the only file operation in this module: it moves what
is marked `discard` into `<folder>/_Duplicates`, or on request sends it to the
desktop trash, and flags the rows deleted. Never an unlink — a review that went
too far is undone from a file manager either way.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .. import filesystem
from ..domain.similarity import cluster, hamming, similarity
from ..scanning import ScanStats, scan
from ..storage import DISCARD, MARKS
from .context import AppContext

log = logging.getLogger(__name__)

DEFAULT_MAX_DISTANCE = 10


def scan_folders(ctx: AppContext, on_progress: Callable[[ScanStats], None] | None = None
                 ) -> ScanStats:
    """Index the duplicate folders into the duplicate index.

    An ordinary scan — the same walker, the same hashing — pointed at a
    different folder set and a different index. Nothing here runs the detector
    or the semantic pass: a duplicate check only ever needs the perceptual hash
    the scan already computes for every image it indexes.
    """
    dupes = ctx.settings.dupes
    dupes.require_folders()
    return scan(
        dupes.as_library(), ctx.dupes.images,
        workers=ctx.settings.workers.scan, on_progress=on_progress,
        # Never re-index what a discard just moved out of the way, or every
        # re-scan would offer the same photo back as a duplicate of the copy
        # that was kept.
        exclude_paths=frozenset({dupes.trash_folder.expanduser().resolve()}),
    )


def groups(ctx: AppContext, max_distance: int = DEFAULT_MAX_DISTANCE,
          limit: int = 200) -> list[dict[str, Any]]:
    """Groups of images that look alike, largest group first.

    Each image carries how alike it is to the group's suggested keeper, as a
    percentage, and whatever mark it already has — so a review interrupted
    halfway resumes where it stopped instead of starting over.

    A byte-identical pair lands in a group here too — identical bytes hash to
    an identical phash, so a distance of 0 always matches — making this a
    superset of `insights.duplicates`, not a competing notion of "duplicate".
    """
    rows = {row["id"]: row for row in ctx.dupes.reporting.phashes()}
    hashes = {image_id: int(row["phash"], 16) for image_id, row in rows.items()}
    dismissed = ctx.dupes.dupes.dismissed_pairs()
    marks = ctx.dupes.marks.all()
    result = []
    for image_ids in cluster(hashes, max_distance, excluded_pairs=dismissed):
        result.extend(_split_around_keepers(image_ids, rows, hashes, marks, max_distance))
    # Biggest first, then tightest — the groups worth looking at before the ones
    # the threshold merely swept up.
    result.sort(key=lambda g: (len(g["images"]), g["similarity"]), reverse=True)
    return result[:limit]


def _split_around_keepers(image_ids: list[int], rows: dict[int, Any], hashes: dict[int, int],
                          marks: dict[int, str], max_distance: int) -> list[dict[str, Any]]:
    """Turn one transitive cluster into groups that keep the threshold's promise.

    `cluster` is deliberately transitive: A close to B and B close to C puts all
    three together even when A and C are nothing like each other. That finds
    candidates well and describes them badly — a group built by chaining can
    hold two photos twice `max_distance` apart, so "max distance 10" would sit
    above a pair labelled 56% alike, which is not a threshold anyone can reason
    about.

    So the chain is peeled instead: take the best copy, keep everything within
    `max_distance` *of it*, and repeat on whatever is left. Every photo in a
    group is then within the threshold of the one photo it is compared against
    on screen, and the percentages mean what the number above them says.
    """
    remaining = [_describe(rows[i], marks) for i in image_ids]
    groups: list[dict[str, Any]] = []
    while len(remaining) > 1:
        # Suggested only. Every percentage in the group is quoted against this
        # one, so the number a person reads answers "how close is it to the one
        # I would otherwise keep" rather than to some arbitrary member.
        keeper = max(remaining, key=_keeper_score)
        near, rest = [], []
        for image in remaining:
            image["similarity"] = similarity(hashes[keeper["id"]], hashes[image["id"]])
            (near if hamming(hashes[keeper["id"]], hashes[image["id"]]) <= max_distance
             else rest).append(image)
        if len(near) > 1:
            near.sort(key=lambda i: (-i["similarity"], i["filename"]))
            groups.append({
                "keep": keeper["id"],
                "images": near,
                # The loosest pairing against the keeper — now guaranteed to be
                # no further away than `max_distance`.
                "similarity": min(image["similarity"] for image in near),
            })
        elif not rest:
            break
        remaining = rest
    return groups


def _describe(row, marks: dict[int, str]) -> dict[str, Any]:
    """One image's worth of what the review UI needs to render and compare it."""
    return {
        "id": row["id"], "path": row["path"], "filename": row["filename"],
        "size": row["size"], "width": row["width"], "height": row["height"],
        "taken_at": row["taken_at"], "mark": marks.get(row["id"]),
    }


def _keeper_score(image: dict[str, Any]) -> tuple[int, int]:
    """Highest resolution first, then largest file — the suggested keeper
    the UI shows pre-selected. The user can always pick a different one."""
    return ((image["width"] or 0) * (image["height"] or 0), image["size"] or 0)


def mark(ctx: AppContext, image_ids: list[int], mark: str | None) -> int:
    """Record what a person decided about these photos, and do nothing else.

    `mark` is `keep`, `discard`, or `None` to go back to undecided. No file is
    moved, copied or deleted — the whole point of a mark is that it is a note
    to yourself that survives closing the tab.
    """
    if mark is None:
        return ctx.dupes.marks.clear(image_ids)
    return ctx.dupes.marks.set(image_ids, mark)


def marked(ctx: AppContext, mark: str) -> list[int]:
    """Every image id currently marked `mark` — what a later "now actually do
    something about these" step would read, once there is one."""
    if mark not in MARKS:
        raise ValueError(f"unknown mark {mark!r}; expected one of {', '.join(MARKS)}")
    return sorted(i for i, value in ctx.dupes.marks.all().items() if value == mark)


def dismiss(ctx: AppContext, image_ids: list[int]) -> None:
    """Record every pair within `image_ids` as "not duplicates" — the
    grouper will not propose any of them together again."""
    ctx.dupes.dupes.dismiss(image_ids)


def discard_marked(ctx: AppContext, delete: bool = False) -> dict[str, Any]:
    """Act on every photo marked `discard`, and return what happened.

    Two destinations, and the caller picks: by default each photo *moves* to
    `<folder>/_Duplicates`, keeping the folder structure it had so two
    same-named files cannot land on top of each other. With `delete`, each goes
    to the desktop trash instead — for people who would only empty the
    duplicates folder by hand afterwards anyway. Neither unlinks anything, so
    an over-eager review is undone from a file manager either way.

    Only what is marked, and only what is still marked when the button is
    pressed — a mark cleared in the meantime is not acted on. The index row is
    flagged deleted rather than removed, so a re-scan does not index the file
    back out of the trash folder and propose it all over again.
    """
    dupes = ctx.settings.dupes
    dupes.require_folders()
    destination = "the desktop trash" if delete else str(dupes.trash_folder)
    ids = marked(ctx, DISCARD)
    if not ids:
        return {"count": 0, "errors": 0, "deleted": delete, "destination": destination}

    count, errors = 0, 0
    for image_id in ids:
        path = ctx.dupes.images.path_of(image_id)
        if path is None or not filesystem.exists(path):
            # Already gone: still flag it, so it stops being offered.
            ctx.dupes.images.mark_deleted(image_id)
            continue
        try:
            if delete:
                filesystem.trash(path)
            else:
                relative = filesystem.relative_to_any(path, dupes.folders)
                filesystem.move(path, dupes.trash_folder / relative, disambiguate_with=image_id)
        except OSError as exc:
            errors += 1
            log.warning("could not discard %s: %s", path, exc)
            continue
        ctx.dupes.images.mark_deleted(image_id)
        count += 1
    ctx.dupes.marks.clear(ids)
    return {"count": count, "errors": errors, "deleted": delete, "destination": destination}
