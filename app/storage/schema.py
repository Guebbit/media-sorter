"""The index layout, as data.

Bumping `SCHEMA_VERSION` is the whole migration story: the index is a cache, not
a system of record, so an old one is thrown away and rebuilt on the next open
rather than half-upgraded (see `engine.Engine._rebuild_if_stale`). Nothing here
is worth a migration — a rescan re-derives all of it.
"""

from __future__ import annotations

SCHEMA_VERSION = "7"

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id           INTEGER PRIMARY KEY,
    path         TEXT    NOT NULL UNIQUE,
    filename     TEXT    NOT NULL,
    root         TEXT    NOT NULL,
    hash         TEXT,
    phash        TEXT,
    size         INTEGER,
    mtime        REAL,
    width        INTEGER,
    height       INTEGER,
    format       TEXT,
    taken_at     TEXT,
    category     TEXT,
    action       TEXT,
    needs_review INTEGER NOT NULL DEFAULT 0,
    detect_state   INTEGER NOT NULL DEFAULT 0,
    adjudicate_state INTEGER NOT NULL DEFAULT 0,
    analyze_state INTEGER NOT NULL DEFAULT 0,
    missing      INTEGER NOT NULL DEFAULT 0,
    deleted      INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    first_seen   REAL    NOT NULL,
    updated_at   REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_detect  ON images(detect_state)   WHERE missing = 0;
CREATE INDEX IF NOT EXISTS idx_images_adjudicate ON images(adjudicate_state) WHERE missing = 0;
CREATE INDEX IF NOT EXISTS idx_images_analyze ON images(analyze_state) WHERE missing = 0;
CREATE INDEX IF NOT EXISTS idx_images_hash    ON images(hash);
CREATE INDEX IF NOT EXISTS idx_images_phash   ON images(phash);
CREATE INDEX IF NOT EXISTS idx_images_cat     ON images(category);
CREATE INDEX IF NOT EXISTS idx_images_root    ON images(root);

CREATE TABLE IF NOT EXISTS detections (
    id         INTEGER PRIMARY KEY,
    image_id   INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    class      TEXT    NOT NULL,
    confidence REAL    NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    model      TEXT,
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_det_image ON detections(image_id);
CREATE INDEX IF NOT EXISTS idx_det_class ON detections(class);

-- The second opinion, kept beside the raw detections rather than folded into
-- them: overwriting a detector's own numbers would make `recheck` unable to
-- tell what was seen from what was later argued, and re-deciding from stored
-- evidence is the whole reason the evidence is stored.
CREATE TABLE IF NOT EXISTS adjudications (
    id         INTEGER PRIMARY KEY,
    image_id   INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    class      TEXT    NOT NULL,
    verdict    TEXT    NOT NULL,
    confidence REAL    NOT NULL DEFAULT 0,
    model      TEXT,
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_adj_image ON adjudications(image_id);

CREATE TABLE IF NOT EXISTS metadata (
    image_id   INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    json       TEXT    NOT NULL,
    model      TEXT,
    created_at REAL    NOT NULL
);

-- Every path `apply` has written into the output tree, so a later run can
-- recognise its own work and prune what no longer belongs. The path is the
-- whole record: what produced it is the current ruleset's business, not
-- history's, and re-deriving it is exactly what `apply` already does.
CREATE TABLE IF NOT EXISTS links (
    id         INTEGER PRIMARY KEY,
    image_id   INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    link_path  TEXT    NOT NULL UNIQUE,
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_links_image ON links(image_id);

CREATE TABLE IF NOT EXISTS action_log (
    id         INTEGER PRIMARY KEY,
    image_id   INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    action     TEXT    NOT NULL,
    detail     TEXT,
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_action_image ON action_log(image_id);

-- A user's "these two are not duplicates" verdict on one pair, so the
-- near-duplicate grouper stops proposing it again. `image_id_a` is always the
-- smaller id — enforced by the repository, not here, so the same pair can
-- never be recorded both ways round.
CREATE TABLE IF NOT EXISTS dupe_dismissals (
    image_id_a INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    image_id_b INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    created_at REAL    NOT NULL,
    PRIMARY KEY (image_id_a, image_id_b)
);

-- What a person decided about one photo while reviewing near-duplicates:
-- `keep` or `discard`. A decision only — nothing here moves, deletes or
-- touches a file. Unmarked photos simply have no row, so "not looked at yet"
-- and "deliberately left alone" stay distinguishable.
CREATE TABLE IF NOT EXISTS dupe_marks (
    image_id   INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    mark       TEXT    NOT NULL,
    created_at REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
