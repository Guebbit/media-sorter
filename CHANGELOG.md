# Changelog

Notable changes, newest first. The index is a cache and the rules file is
plain data, so "how to upgrade" is almost always "nothing" — anything a change
invalidates is re-derived by the next scan. Where that is not true, it says so.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Videos are sorted by what is in them.** YOLO runs on
  `MEDIASORT_DETECT_VIDEO_FRAMES` stills (8 by default) sampled across each
  file, so a clip of your cat lands in `Cat/` next to the photos of it. The
  frames collapse to one verdict per file — per class, the frame that saw the
  most of it wins, ties to the more confident frame — which keeps `min_count`
  conditions meaningful. Every video still carries the `video` class on top of
  whatever was found, so a `{"class": "video"}` rule below the real ones is the
  fallback for clips nothing was recognised in. `MEDIASORT_DETECT_VIDEO_FRAMES=0`
  goes back to matching videos by extension alone. New module `app/video.py`
  (frame sampling, the only place OpenCV is imported, and lazily),
  `Detector.detect_video`, `domain.detection.merge_frames`.
- **Near-duplicate finder**, at `/duplicates` in the web UI. Its own folders,
  its own index and its own scan — checking a folder for duplicates does not
  add it to the sorting library or run the detector over it. Photos are grouped
  by perceptual hash (dHash, Hamming distance) and scored as a percentage
  against the group's suggested keeper, so a re-save, a resize or a format
  change is caught where a checksum is blind. Marking `keep`/`discard` writes
  nothing to disk; acting on the marks is a separate press that either moves
  them to `<folder>/_Duplicates` or sends them to the desktop trash
  (`send2trash`), never an unlink. "Not duplicates" dismisses a pair so the
  grouper stops proposing it. New `app/domain/similarity.py`,
  `app/services/duplicates.py`, `app/storage/dupes.py`,
  `app/interfaces/web/static/dupes.html`.
- A perceptual hash for every indexed image, computed in the same decode the
  scanner already does for dimensions and EXIF (`imaging.inspect`, replacing
  `imaging.probe`). Stored as `images.phash`.
- **Run everything** in the web UI header: scan, examine and sort as one job on
  the server, so closing the browser halfway does not stop the chain. It renames
  itself per phase, and the step cards below light up as if each button had been
  pressed in turn.
- Image and thumbnail routes for the web UI, for both indexes — the same id
  means a different photo in each, so they are separate endpoints on purpose.
- `docs/CLI.md`: every command and option, with recipes, exit codes and
  troubleshooting.
- `mediasort config set --save-index / --no-save-index`, plus the "Remember this
  library" and "Remember the duplicate scan" toggles in Settings.
- Rules parsing rejects unknown fields — in the rules file, in a rule, and in a
  `class` condition — instead of ignoring them. A typo in a threshold used to be
  a rule that quietly did something other than what its file said.

### Changed

- **The index is in memory by default and nothing is written.** It only pays for
  itself when the same folder is worked on twice, so pointing the tool at a
  folder once now leaves no file behind. Turn it on with `SAVE_INDEX`, and it
  lives *inside the library it describes* at `<photo folder>/.mediasort/index.db`
  — one index per library, travelling with the drive, unable to contaminate
  another folder's work. Setting `MEDIASORT_DB` still names a file explicitly and
  always saves. In-memory engines use a per-engine shared-cache URI (one
  keepalive connection, `read_uncommitted` for the UI's stats polling, which
  shared cache would otherwise fail outright with a table lock).
- An index from an older schema is **thrown away and rebuilt** on open rather
  than refused. Everything in it is re-derivable, and a cache that argues with
  you at the moment you wanted to sort photos is a cache doing its job badly.
  Schema is now v7 (from v4).
- **`move` and `delete` rules run unprompted.** The ruleset is the confirmation:
  a rule that says `delete` deletes, and nothing downstream second-guesses it.
  `mediasort doctor` now *warns*, naming every rule that consumes originals, and
  `apply --dry-run` still lists every source and destination first.
- The photo and output folders are ordinary settings again, read from all three
  layers like everything else. `MEDIASORT_INPUT_FOLDERS` /
  `MEDIASORT_OUTPUT_FOLDER` are no longer ignored.
- A settings field the tool derived (the output folder) is shown empty with a
  placeholder describing it, so "left alone" and "typed by hand" stay
  distinguishable and clearing the box is the way back to the default.
- Class-band resolution lives in one place: `ClassBands` (with `for_class` and
  `unsettled`) and `merge_overrides`, shared by the decision engine, the
  adjudicator and the rule set, each of which carried its own copy of the same
  arithmetic and the same default-band fallback.
- `imaging.to_jpeg_bytes` is the shared downscale path (`to_jpeg_base64` now
  wraps it), and it handles videos by way of a frame from halfway in — which is
  also all Ollama sees of a video, for both the description pass and a second
  opinion.
- `AppContext.reloaded` owns what a settings reload rebuilds; front ends only
  decide when. Both indexes are reopened only if they actually moved.
- `opencv-python` and `send2trash` are declared dependencies. OpenCV arrives with
  ultralytics anyway, but this code imports `cv2` itself; `send2trash` is
  imported at the call site, so a missing install is a message on the duplicates
  page's delete button rather than a broken start-up.

### Removed

- `--yes` / `-y` on `mediasort apply` and `mediasort run`, and the confirm
  checkbox in the web UI, along with the `permitted` plumbing behind them. (The
  `--yes` on `clean` and the other destructive maintenance commands stays.)
- The `version` field in `rules.json`. It is written no longer, and a file that
  still carries one is now rejected by `mediasort rules validate` — delete the
  line, or re-run `mediasort rules init`.
- `IncompatibleIndex`, replaced by the silent rebuild above.
- The `links.mode` column and `verify_links`' "broken" state: nothing creates a
  symlink any more, so a copy either exists or it does not.
- `overrides.NO_ENVIRONMENT` / `Editable.from_environment`, the `number` field
  kind, and `_Layers.float_` — no editable setting needs any of them now that
  confidence bands are per rule.
- `trash_folder` from the web UI's config payload; nothing on the page read it.

### Fixed

- **The detector was fed RGB arrays.** ultralytics reads a numpy array as
  OpenCV wrote it — BGR — so every photo went through the model with red and
  blue swapped, costing accuracy for nothing. Frames now arrive in BGR from
  both paths. Re-run `mediasort detect` on an existing index to benefit.
- An editable setting without a reader in `services.configuring` surfaced as a
  `KeyError` from whichever endpoint happened to render the form first; the two
  lists are now checked against each other on import.
