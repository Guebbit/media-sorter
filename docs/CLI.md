# The command line

Everything the web UI does, and several things it does not. The UI is the
comfortable way to edit rules and watch a run; the CLI is the one that scripts,
runs over SSH, and tells you exactly what went wrong.

Both front ends write the same files and read the same index, so you can start a
run here and watch it there, or save a folder in the browser and scan it from a
terminal. Nothing is duplicated between them except the buttons.

- [Getting the command](#getting-the-command)
- [The five-minute path](#the-five-minute-path)
- [Where settings come from](#where-settings-come-from)
- [The commands](#the-commands)
  - [Indexing and processing](#indexing-and-processing)
  - [Acting on the results](#acting-on-the-results)
  - [Rules](#rules)
  - [Configuration](#configuration)
  - [Looking at the index](#looking-at-the-index)
  - [Repairing and throwing away](#repairing-and-throwing-away)
  - [The web UI](#the-web-ui)
- [Environment variables](#environment-variables)
- [Recipes](#recipes)
- [Interrupting, resuming, exit codes](#interrupting-resuming-exit-codes)
- [Troubleshooting](#troubleshooting)

## Getting the command

`mediasort` is a console script installed into the project's virtualenv:

```bash
make install            # or make install-cpu — once
source .venv/bin/activate
mediasort --help
```

Without activating the virtualenv, name it in full — `.venv/bin/mediasort run`.
Both are the same program; `make run`, `make scan` and friends are one-line
wrappers around exactly these commands and exist only so you do not have to
activate anything.

Every command and sub-command takes `--help`, and it is generated from the code
rather than written twice, so it is never out of date:

```bash
mediasort apply --help
mediasort rules init --help
```

Most commands also take `--verbose` / `-v`, which drops the log level to DEBUG
for that run. It is the first thing to try when something is quietly doing
nothing.

## The five-minute path

```bash
mediasort config set --input ~/Pictures        # the one thing it cannot guess
mediasort rules init --classes cat,dog,video   # nothing else seeds this for you
mediasort doctor                               # GPU, weights, Ollama, rules, paths
mediasort run                                  # scan -> detect -> analyze -> apply
mediasort stats                                # what came out
```

`run` is the whole pipeline in one command, and it is the one to reach for by
default. The individual stage commands below exist for when you want to pay for
one pass at a time, or re-run one of them.

Results land in the output folder — by default `<your first photo
folder>/Sorted`, one subfolder per rule that matched.

## Where settings come from

Three layers, lowest first:

1. **built-in defaults** — a complete working run on their own;
2. **`.env`** in the project folder, and the real environment, which wins over
   the file;
3. **`mediasort config set`** (and the UI's Settings tab), stored in
   `data/settings.json`, which wins over both.

`mediasort config show` prints every editable setting with the layer it came
from, so "why is it using that model" is one command rather than a hunt.

There are three places to say where the photos are, and a saved choice beats an
exported variable:

```bash
mediasort config set --input ~/Pictures     # saved; every later command sees it
mediasort scan --input /mnt/usb-drive       # this run only, nothing written
# ...or the folder fields in the web UI, which write the same file config set does
```

`--input` is repeatable and accepts a comma-separated list, so `-i ~/Photos -i
/mnt/nas/pics` and `--input ~/Photos,/mnt/nas/pics` are the same thing. Folders
are validated before they are stored, so a typo is refused rather than saved.

There is **no default for the input folders at all**: with none set, `scan` says
so and stops, and `doctor` reports it as a failing check.

## The commands

### Indexing and processing

#### `mediasort scan`

Walks the input folders and indexes new or changed files. Safe to re-run at any
time, and cheap: a file whose size and mtime match the index is skipped without
being opened, which is what keeps a re-scan of a large library to a couple of
minutes. Only new or changed files get a SHA256, a perceptual hash and an
image-header probe.

| option | meaning |
| --- | --- |
| `--input`, `-i FOLDER` | Scan these folders instead of the saved ones. Repeatable. This run only. |
| `--verbose`, `-v` | DEBUG logging. |

A file that has changed resets its own pipeline state and discards its stale
detections. Files that have vanished are flagged `missing`, never deleted — an
unplugged drive costs you nothing.

One subtlety worth knowing: scanning with `--input` is treated as a *partial*
scan, so files outside those folders are not marked missing. A scan without
`--input` is authoritative over the whole saved library.

#### `mediasort detect`

The YOLO pass, on every image that has not been through it yet. Prints the device
it chose, the weights, the video frame count and — from your rules — the list of
classes it is actually looking for.

| option | meaning |
| --- | --- |
| `--limit`, `-n N` | Stop after N images (0 = all). Good for a first taste on a big library. |
| `--verbose`, `-v` | DEBUG logging. |

Detections are stored down to the lowest `min_review_confidence` any rule asks
for, not the higher threshold that auto-passes them. Those extra rows are what
make `recheck` able to answer a threshold change without the GPU.

#### `mediasort adjudicate`

The second opinion: for each borderline detection whose class has
`ollama_review` on, the vision model is asked one question — *is this really
there?* — and answers `present`, `absent` or `unsure`. Prints what it settled at
the end.

| option | meaning |
| --- | --- |
| `--limit`, `-n N` | Stop after N images (0 = all). |
| `--verbose`, `-v` | DEBUG logging. |

If no rule asks for a second opinion, it says so and does nothing. `unsure` is a
real answer and changes nothing, which leaves the photo exactly as uncertain as
it was — that is what `_Review` is for.

#### `mediasort analyze`

The semantic pass: a description from the vision model, stored as JSON and
searchable with `mediasort search`. Runs **only** on images the rules did not
`ignore`, which on a typical library skips 80-90% of the work.

| option | meaning |
| --- | --- |
| `--limit`, `-n N` | Stop after N images (0 = all). |
| `--verbose`, `-v` | DEBUG logging. |

#### `mediasort run`

Scan, then the three passes (detect and analyze run in parallel), then apply the
rules. One progress bar per stage; the downstream totals grow as the detector
discovers work for them.

| option | meaning |
| --- | --- |
| `--input`, `-i FOLDER` | Folders for this run only. Repeatable. |
| `--output FOLDER` | Write the sorted tree here for this run only. |
| `--no-analyze` | Skip the semantic pass — detection and rules only. |
| `--no-apply` | Do the passes but do not touch any files. |
| `--skip-scan` | Reuse the existing index; do not walk the filesystem. |
| `--verbose`, `-v` | DEBUG logging. |

`--no-analyze` is the fast configuration: the detector plus your rules is enough
to sort a library, and the description pass is what makes it *searchable*. It
does not disable adjudication — that still runs if a rule asks for it.

If you interrupt the run, the actions are not applied; the passes are still saved.

#### `mediasort recheck`

Re-decides every image from the detections and verdicts already in the index. No
GPU, no Ollama, no network, and it takes seconds.

| option | meaning |
| --- | --- |
| `--verbose`, `-v` | DEBUG logging. |

**This is the command after editing rules.** Neither model runs twice. The usual
loop is `recheck` → `apply --dry-run` → `apply`.

### Acting on the results

#### `mediasort apply`

Executes what the rules decided: builds the output tree, runs the moves and the
deletions. Idempotent — it computes the whole desired tree from the database
first, prunes what no longer belongs, then creates what is missing, so running it
twice creates nothing the second time.

| option | meaning |
| --- | --- |
| `--dry-run`, `-d` | Print every source and destination without touching anything. |
| `--output FOLDER` | Build the tree here instead of the saved folder. This run only. |
| `--no-prune` | Leave links that no longer match where they are. |
| `--verbose`, `-v` | DEBUG logging. |

**Run `--dry-run` first the first time, and after any rule change.** `move` and
`delete` rules act on your originals on every ordinary run, with nothing to
confirm — the ruleset is the only thing standing between a rule and your
library. `delete` always moves to the trash folder rather than unlinking, and
every deletion is recorded in `mediasort history`.

`--no-prune` matters when you pass a one-run `--output`: without it, `apply`
prunes the links it recorded under the previous output folder.

### Rules

The rules file lives at `MEDIASORT_RULES` (`./data/rules.json`). Nothing creates
it for you. Full syntax is in the README; these are the commands around it.

#### `mediasort rules show`

The active rules in evaluation order, with which actions consume originals, the
file's path, and the classes the detector will be asked for. First match wins, so
the order in this table *is* the policy.

#### `mediasort rules init`

Writes a starter file: one combined rule, one rule per class, one catch-all.

| option | meaning |
| --- | --- |
| `--classes cat,dog,bird` | Comma-separated classes to seed. |
| `--force` | Overwrite an existing rules file. |

Refuses to clobber an existing file without `--force`, and exits 1 if it would
have to.

#### `mediasort rules validate`

Checks the file parses, and that every action named is one that exists.

| option | meaning |
| --- | --- |
| `--check-classes` | Also confirm the detector can actually find every class used. Loads torch, so it is slower. |

Exits 1 with the reason on a bad file. Worth putting in a pre-commit hook if you
keep your rules in git.

#### `mediasort rules actions`

The four actions a rule can use, and what each one does to the original.

#### `mediasort rules classes`

Every class a rule can match: the configured detector's (80 COCO classes with the
default weights), plus the `video` pseudo-class. Loads torch.

### Configuration

#### `mediasort config show`

Every editable setting, its current value, and which layer supplied it —
`settings`, `environment` or `default`.

#### `mediasort config set`

Saves settings to `data/settings.json`, the same file the Settings tab writes.
Takes effect for every later command with no restart.

| option | meaning |
| --- | --- |
| `--input`, `-i FOLDER` | Folders to index. Repeatable; **replaces** the saved list. |
| `--output FOLDER` | Where the sorted tree goes. Clearing it restores the default. |
| `--ollama-url URL` | The Ollama you already run. |
| `--ollama-model NAME` | Any vision-capable model installed there. |
| `--save-index` / `--no-save-index` | Keep the index inside the library so a second run does not start over. Off by default. |

With no options at all it fails rather than doing nothing silently. It re-reads
the file afterwards and prints what is now in force, rather than echoing what you
typed.

On `--save-index`: by default the index is held in memory for the run and
dropped when the command ends — point the tool at a folder, sort it, walk away,
and nothing is written into your photos. Turn it on for a library big enough
that losing a half-finished detection run would hurt; the index then lives at
`<photo folder>/.mediasort/index.db`, one per library, and travels with the
drive.

#### `mediasort config unset KEY...`

Forgets saved settings, handing them back to `.env` or the default. Keys are
`INPUT_FOLDERS`, `OUTPUT_FOLDER`, `OLLAMA_URL`, `OLLAMA_MODEL`, `SAVE_INDEX` —
case-insensitive, and several at once is fine:

```bash
mediasort config unset output_folder ollama_model
```

### Looking at the index

None of these change anything.

#### `mediasort stats`

Per-stage progress, category counts, action tallies, and how many photos are
flagged for review. The quickest answer to "what is left to do".

#### `mediasort doctor`

The pre-flight check: configuration and where each value came from, the photo
folders, the rules file, GPU and torch device, model weights, Ollama
connectivity, and **every rule that consumes originals**. That last one is the
warning you get before a `move` or `delete` rule runs, so read it after editing
rules.

#### `mediasort verify`

Confirms the originals the index knows about, and the links it created, are still
on disk. Run it after moving a drive around.

#### `mediasort duplicates`

Files sharing a SHA256 — byte-identical copies.

| option | meaning |
| --- | --- |
| `--limit`, `-n N` | Groups to show (default 20). |

This is exact-match only. Finding photos that merely *look* the same — a re-save,
a resize, a different format — is the web UI's **Duplicates** tab, which compares
perceptual hashes and lets you mark keepers; there is no CLI equivalent, and by
design nothing there moves or deletes a file.

#### `mediasort history`

Recent actions that moved or removed an original, newest first. This is the
record of everything irreversible the tool has done.

| option | meaning |
| --- | --- |
| `--limit`, `-n N` | Entries to show (default 30). |

#### `mediasort search TERM`

Free-text substring search over the stored analysis metadata — breeds,
environments, counts, whatever your vision model filled in. Needs `analyze` to
have run.

| option | meaning |
| --- | --- |
| `--limit`, `-n N` | Results to show (default 50). |

```bash
mediasort search "maine coon"
mediasort search beach -n 200
```

For anything more structured than a substring, query the SQLite file directly —
the schema is in the README, and `json_extract` over the `metadata` table is
where the interesting questions live.

#### `mediasort export`

Dumps the index, analysis metadata included, to JSON or CSV.

| option | meaning |
| --- | --- |
| `--out`, `-o PATH` | Destination. Defaults to `<output folder>/export.<format>`. |
| `--format`, `-f json\|csv` | Default `json`. |
| `--category NAME` | Only images whose matched rule was this one. |

```bash
mediasort export -f csv --category Cat -o ~/cats.csv
```

### Repairing and throwing away

#### `mediasort retry`

Puts images that errored back in the queue.

| option | meaning |
| --- | --- |
| `--stage detect\|adjudicate\|analyze` | Just this stage. Default: all of them. |

The usual cause of a batch of errors is Ollama having been down; start it and
`retry`.

#### `mediasort reset STAGE`

Marks a stage pending again so it reruns on **every** image. `STAGE` is
`detect`, `adjudicate`, `analyze` or `all`.

| option | meaning |
| --- | --- |
| `--yes`, `-y` | Do not ask for confirmation. |

Asks before proceeding unless you pass `-y`. Resetting `adjudicate` (or `all`)
also re-decides from the raw detections, because an `absent` verdict cleared the
review flag and that flag is the queue.

This is expensive — it throws away GPU and Ollama work. If what changed is your
*rules*, you want `recheck`, not `reset`.

#### `mediasort clean`

Removes generated output and stale rows. **Never touches originals.** Requires at
least one flag.

| option | meaning |
| --- | --- |
| `--links` | Delete the whole output tree. Asks first unless `--yes`. |
| `--missing` | Drop index rows whose file is no longer on disk. |
| `--vacuum` | Compact the index. |
| `--yes`, `-y` | Do not ask. |

`clean --links` followed by `apply` rebuilds the output tree from scratch, which
is the fix for a tree you have been rearranging by hand.

### The web UI

#### `mediasort web`

Serves the rules editor and control panel, and fails fast on a broken rules file
rather than starting and showing you an error page.

| option | meaning |
| --- | --- |
| `--port`, `-p N` | Override the configured port (default 8765). |
| `--verbose`, `-v` | DEBUG logging. |

It is **unauthenticated by design** — it edits your settings and can delete your
files. Keep `MEDIASORT_WEB_HOST` on `127.0.0.1`; if you need it from another
machine, tunnel over SSH rather than binding it wide.

## Environment variables

All prefixed `MEDIASORT_`, all read from `.env` in the project folder or from the
real environment (which wins). Everything here has a working default; the four in
`.env.example` are the ones worth touching.

| Variable | Default | What it does |
| --- | --- | --- |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Your Ollama. Also settable in the UI / `config set`. |
| `OLLAMA_MODEL` | `llava-llama3` | Vision model for both Ollama passes. Also in the UI. |
| `DETECT_MODEL` | `yolo11m.pt` | `yolo11n` (fast) … `yolo11x` (accurate). |
| `DETECT_VIDEO_FRAMES` | `8` | Frames sampled per video. `0` sorts videos by extension only. |
| `DETECT_BATCH` | `16` | Images per inference batch. Lower it if the GPU runs out of memory. |
| `DETECT_IMGSZ` | `640` | Inference image size. |
| `DETECT_DEVICE` | `auto` | `cuda`, `cpu`, or a device index. |
| `MODEL_SEARCH_PATHS` | — | Extra comma-separated folders to look for weights in. |
| `MODELS_DIR` | `data/models` | Where weights are downloaded to. |
| `RULES` | `data/rules.json` | The rules file. |
| `DB` | — | An explicit index file. Naming one is itself a request to save it. |
| `SETTINGS` | `data/settings.json` | The overlay `config set` and the UI write. |
| `ENV_FILE` | — | Point at a specific `.env`; a missing file loads nothing. |
| `EXTENSIONS` | images + video | Which files count as the library. |
| `EXCLUDE_DIRS` | `.git,node_modules,@eaDir,output,Output,Sorted` | Visible folders to skip. Hidden ones are always skipped. |
| `FOLLOW_SYMLINKS` | `0` | Whether the scanner follows symlinks. |
| `TRASH_FOLDER` | `<output>/_Trash` | Where `delete` rules put originals. |
| `OLLAMA_TIMEOUT` | `180` | Seconds per request. |
| `OLLAMA_NUM_CTX` | `4096` | Context window. |
| `OLLAMA_MAX_EDGE` | `1024` | Images are downscaled to this before encoding. |
| `OLLAMA_KEEP_ALIVE` | `10m` | How long Ollama holds the model in memory. |
| `WORKERS_SCAN` | `4` | Scanner threads. |
| `WORKERS_DETECT` | `4` | Threads decoding images for the detector. |
| `WORKERS_ANALYZE` | `2` | Concurrent Ollama requests. |
| `WEB_HOST` | `127.0.0.1` | Where `web` binds. |
| `WEB_PORT` | `8765` | Port for `web`. |
| `LOG_LEVEL` | `INFO` | `--verbose` forces DEBUG for one run. |
| `INPUT_FOLDERS` | — | **Ignored.** Use `config set --input` or `--input`. |
| `OUTPUT_FOLDER` | — | **Ignored.** Use `config set --output` or `--output`. |

Confidence thresholds and the second-opinion toggle are **not** environment
variables — they are per class condition in a rule (`min_confidence`,
`min_review_confidence`, `ollama_review`), defaulting to 0.65 / 0.35 / on.

Because the environment is a layer, a one-off override is just a prefix:

```bash
MEDIASORT_OLLAMA_MODEL=qwen2.5-vl mediasort analyze
MEDIASORT_DETECT_BATCH=4 mediasort detect        # a GPU short on memory
```

## Recipes

**Try it on a folder without committing to anything**

```bash
mediasort run --input /mnt/usb --output /tmp/sorted --no-analyze
```

Nothing is saved: not the folders, not the index. Detection only, so no Ollama
needed.

**See what your rules would do before they do it**

```bash
mediasort recheck            # re-decide from what is already in the index
mediasort apply --dry-run    # every source and destination, nothing written
mediasort apply
```

**Iterate on rules without paying for the GPU again**

Edit `data/rules.json` (or use the UI's editor), then:

```bash
mediasort rules validate && mediasort recheck && mediasort apply --dry-run
```

Detections and verdicts are both reused. This is the loop the storage layout
exists to make possible.

**A first pass on a huge library**

```bash
mediasort scan
mediasort detect --limit 500     # see whether the rules are right
mediasort stats
```

**Overnight, on a big library, resumably**

```bash
mediasort config set --save-index
mediasort run
```

Kill it whenever. Run the same command again and it picks up from the database.

**Ollama was down for half the run**

```bash
mediasort retry --stage analyze
mediasort analyze
```

**Move the whole library to a different sorting scheme**

```bash
mediasort clean --links --yes    # tear down the output tree
mediasort recheck                # re-decide with the new rules
mediasort apply
```

Note that this rebuilds *copies* only. A `move` rule consumed its original, so
re-categorising a moved photo means moving it again, not undoing anything.

**Check the setup after changing anything important**

```bash
mediasort doctor
```

## Interrupting, resuming, exit codes

**Ctrl-C once** finishes the batch in flight and exits cleanly, with progress
saved. **Twice** forces it. There is no separate `resume` command because there
is no separate resume path — every command is resumable by construction, and
re-running `run` continues from the database. Nothing is processed twice unless
the file itself changed.

That resumability depends on the index existing between runs, which is
`config set --save-index`. With the default in-memory index, a Ctrl-C costs you
the run.

Exit codes:

| code | meaning |
| --- | --- |
| `0` | Success. |
| `1` | A validation failure with a printed reason — a bad rules file, `rules init` refusing to clobber. |
| `2` | A configuration or usage error: no photo folders, an unknown setting key, a missing dependency. |
| `130` | A forced exit from a second Ctrl-C. |

Errors print as a single red line rather than a traceback, so a failure in a
script is readable in a log.

**Run one mediasort at a time per index.** Each stage resets rows left `running`
at startup, which assumes any previous run is dead rather than concurrent. The
web UI runs one job at a time by construction, but a CLI command started
alongside it is not coordinated with it.

## Troubleshooting

**`no photo folders configured`** — nothing has said where the photos are. `mediasort
config set --input ~/Pictures`, or pass `--input` for one run. This is deliberate:
there is no default and no environment variable, because a silently redirected
scan is worse than an error.

**`.venv/bin/pip: No such file or directory` after moving the project** — a
virtualenv is not relocatable, and the message names the wrong file: the script
is there, the interpreter in its shebang is not. See the README's "If you move or
rename the project folder". Note that `.venv/bin/python -m pytest` keeps working
throughout, so a green test run is not evidence that the install is fine.

**A command seems to do nothing** — it probably has nothing to do, and says so:
"nothing to detect", "nothing to analyze". `mediasort stats` shows what is
actually pending per stage.

**Nothing matched a rule** — the detector ran but found nothing your rules ask
for. Check `mediasort rules show` (which lists the classes being searched for)
against `mediasort rules classes` (which lists what the model can find), and
consider lowering a `min_confidence` and running `recheck`.

**Everything went to `_Review`** — the second opinion could not settle it, or
never ran. `mediasort doctor` will say if Ollama is unreachable; with it down, an
uncertain photo goes straight to `_Review` by design.

**Out of GPU memory** — lower `MEDIASORT_DETECT_BATCH`, or use smaller weights
(`MEDIASORT_DETECT_MODEL=yolo11n.pt`). Detection also falls back to the CPU on
its own, which is correct and slower; `doctor` reports which device torch chose.

**Slow** — the semantic pass is almost always the cost. `run --no-analyze` skips
it entirely; an `ignore` rule that matches most of your library skips it for
those photos, which is where the design expects most of the saving to come from.
