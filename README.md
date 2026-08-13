# mediasort

Offline photo indexer and sorter. Points at a photo library, finds the photos
containing whatever **you** define, and does whatever **you** decide with them —
by default, a sorted tree of copies that never touches the originals. Videos in
the same library are sorted by their content too, from frames sampled across
each one.

Runs directly on your machine against your own Ollama. No cloud, no network at
runtime, nothing to orchestrate.

```
                             ┌── sure ─────────────────┐
Photos ─► Scanner ─► SQLite ─► Detector          Rules engine ─┬─ copy   ─► folder
                       ▲     └ unsure ─► Ollama ──┘  ▲         ├─ move   ─► folder
                       │                 (verdict)   │         ├─ delete ─► trash
                       └──────── Ollama ─────────────┘         └─ ignore
                                (describes what the rules keep)
```

YOLO decides what it can. What it only half-saw goes to a vision model for a
verdict — present, absent, or *unsure*. Only what neither of them can settle
ends up in front of you.

Cats and dogs are just the **default** rules, not the design. A rule says
*what to match* and *what to do about it*; the pipeline itself knows neither.
Edit them in the built-in web UI, or in `rules.json`.

The database is the handoff between every stage, which is what makes the whole
thing resumable: kill it at any point, run the same command again, and it picks
up exactly where it stopped. Nothing is processed twice unless the file itself
changed.

For a longer explanation of *why* it works this way — YOLO, the second
opinion, the rules engine — see [docs/CONCEPTS.md](docs/CONCEPTS.md). For every
command and option, with recipes, see [docs/CLI.md](docs/CLI.md). For a
tour of the code itself, module by module, see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Requirements

Python 3.11+, and an Ollama you already run (only for the optional semantic
pass). An NVIDIA GPU makes detection dramatically faster but is not required.

## Quick start

```bash
make install          # creates .venv, downloads torch (~2.5 GB) — once
source .venv/bin/activate

mediasort config set --input ~/Pictures   # the one thing it cannot guess
mediasort doctor      # check GPU, weights, Ollama, rules, paths
mediasort run         # scan -> detect -> analyze -> apply the rules
```

`make install-cpu` instead of `make install` if you have no NVIDIA GPU.

`make install` also seeds a starter ruleset — `person,cat,dog,video` — so
`make web` and `make run` have something to work with on a fresh clone. It only
writes the file when there is none, so re-running it never touches rules you
have edited. `make seed DEMO_CLASSES=cat,bird,video` starts from your own
classes instead, and the editor changes them at any point.

The folders are the one setting you have to give, and there are three ways to:
`mediasort config set` as above, the Settings tab in the web UI, or `--input` /
`--output` on a single command. Deliberately not `.env` — see
[Configuration](#configuration).

To change what it looks for, or where it looks, open the control panel:

```bash
make web              # http://127.0.0.1:8765
```

Results land in the output folder — by default a `Sorted` folder inside your photo
folder, so they sit next to the photos they came from — one folder per rule that
matched something:

```
output/
├── Cat/
├── Dog/
├── Cat-Dog/
├── video/       # the clips nothing was recognised in — see "Rules" below
└── _Review/     # what neither model could settle — see the doubt rule
```

Every `make` target is a thin wrapper. Activate the virtualenv and the command
is yours directly:

```bash
source .venv/bin/activate
mediasort run
```

`mediasort` lives in `.venv/bin/`, so it is on your PATH only while that
virtualenv is active. To have it everywhere without activating:

```bash
make link             # symlinks .venv/bin/mediasort into ~/.local/bin
make unlink           # removes it again
```

`make link` warns if `~/.local/bin` is not on your PATH, and takes `BINDIR=` for
somewhere else. The symlink points into this folder — see below.

### If you move or rename the project folder

A Python virtualenv is **not relocatable**. Every script `pip` installs into
`.venv/bin/` gets the interpreter's absolute path baked into its first line:

```
#!/home/you/projects/media-sorter/.venv/bin/python3
```

Move or rename the folder and that path stops existing, so *every* `.venv/bin/`
command breaks at once — `make install`, `make test`, `mediasort`, `pip`:

```
make: .venv/bin/pip: No such file or directory
make: *** [Makefile:25: install] Error 127
```

The message is misleading. `.venv/bin/pip` is right there; what is missing is
the **interpreter named in its shebang**. The kernel reports the failure against
the script it was asked to run, not against the interpreter it could not find,
so anything launched this way names the wrong file.

Two ways out.

**Rebuild it** — always correct, but re-downloads torch:

```bash
make uninstall && make install
```

**Or rewrite the paths in place** — seconds, no download:

```bash
find .venv -type f -exec grep -Il "OLD-FOLDER-NAME" {} + \
  | xargs sed -i 's|/home/you/projects/OLD-FOLDER-NAME|/home/you/projects/NEW-FOLDER-NAME|g'
```

`-type f` matters: `.venv/bin/python` and friends are **symlinks**, and `sed -i`
replaces a symlink with a regular file rather than editing through it, which
breaks the virtualenv in a new and less obvious way. `grep -Il` skips binaries
for the same reason. Around 30 files need it — the `bin/` scripts, `pyvenv.cfg`,
and the editable install's `__editable___*_finder.py` and `direct_url.json`.

Either way, re-run `make link` afterwards if you had one: the symlink names the
old path too.

Check it worked before trusting it:

```bash
.venv/bin/pip --version && .venv/bin/mediasort --help
```

One thing to know while diagnosing this: `.venv/bin/python -m pytest` keeps
working the whole time. Naming the interpreter explicitly skips the shebang, so
the test suite passes on a virtualenv whose every console script is broken. A
green test run is not evidence that `make install` works.

## Configuration

### The folders

The folders are ordinary settings, read from the same three layers as
everything else. In practice you will set them one of three ways, and the same
three work for both folders:

```bash
mediasort config set --input ~/Pictures --output ~/Sorted   # saved, applies to everything after
mediasort config show                                       # what is in force, and from where
mediasort config unset OUTPUT_FOLDER                        # back to the default

mediasort scan --input /mnt/usb-drive        # this run only, nothing saved
mediasort run  --input /mnt/usb-drive --output /mnt/usb-sorted
mediasort apply --output /tmp/try-a-layout
```

`--input` is repeatable (`-i a -i b`) and accepts a comma-separated list. The
third place is `mediasort web`, at the top of the **Run** tab and again in
**Settings** — one form over one draft, so an edit started on either is visible
on the other — which writes the same file `config set` does,
`data/settings.json`. A folder is validated before it is stored either way, so a
typo is rejected rather than saved. Clearing the output folder resets it to the
default rather than storing an empty string.

The input folders have no default at all: with none set, `mediasort scan` says so
and stops, and `mediasort doctor` reports it as a failing check.

The output folder **defaults to `<your first photo folder>/Sorted`** — the results
belong with the photos they came from, not in whichever directory you happened to
run the command from. It is a subfolder rather than the photo folder itself on
purpose: a `Sorted` name the scanner already ignores means a re-scan cannot index
its own output, and a `delete` rule cannot hand you its trash folder back as new
photos. If you would rather have `Cat/` and `Dog/` directly among your photos,
say so explicitly:

```bash
mediasort config set --output ~/Pictures     # same folder as the input
```

That is safe too — the scanner skips whatever the output and trash folders resolve
to, wherever you put them.

A one-run `--output` moves where the tree is built, and `apply` prunes links it
recorded under the old one — pass `--no-prune` to leave them alone.

### Everything else

Three layers, lowest first:

1. **built-in defaults** — a complete working run on their own;
2. **`.env`** in the project folder (or the environment itself, which wins over
   the file, so `MEDIASORT_OLLAMA_MODEL=llava:13b mediasort analyze` works);
3. **`mediasort config set` and the Settings tab**, for the handful of settings
   worth changing without restarting — the two folders, the Ollama URL and model,
   the two detector confidence thresholds, and whether the semantic pass runs
   at all.

Layer 3 is stored in `data/settings.json` and **wins** over `.env`; `config show`
and the UI both say which layer each value came from, and either can hand a
setting back to `.env` — `config unset KEY`, or one click. Everything else is
`.env` only.

Paths are ordinary paths on this machine. The tool sees exactly what you see, so
your photo folders are as writable as any other file you own — which is what
makes `delete` rules possible, and why **the ruleset is the only thing standing
between a rule and your originals**. `move` and `delete` run on every ordinary
run, with nothing to confirm: a rule that says `delete` deletes. `delete` always
goes to the trash folder, never a permanent unlink. Preview first —
`mediasort apply --dry-run` lists every source and destination without touching
anything.

`.env.example` is deliberately short: it contains exactly the four settings
below and nothing else, and every one of them works as shipped.

| Variable                        | Default                  | What it does                                             |
| ------------------------------- | ------------------------ | -------------------------------------------------------- |
| `MEDIASORT_OLLAMA_URL`          | `http://127.0.0.1:11434` | Your Ollama. Also in the UI.                             |
| `MEDIASORT_OLLAMA_MODEL`        | `llava-llama3`           | Vision model for both Ollama passes. Also in the UI.     |
| `MEDIASORT_DETECT_MODEL`        | `yolo11m.pt`             | `yolo11n` (fast) … `yolo11x` (accurate).                 |
| `MEDIASORT_DETECT_VIDEO_FRAMES` | `8`                      | Frames sampled per video. `0` sorts videos by extension only. |

The semantic pass always runs — there is no on/off switch for it any more, in
the UI or in `.env`. The confidence band and the second opinion are per rule
now, not global settings: each class condition in a rule (`{"class": "cat", ...}`)
carries its own `min_confidence` (auto-pass threshold), `min_review_confidence`
(the floor of the band below it), and `ollama_review` (whether a borderline hit
of that class is worth asking Ollama about at all) — set them in the rule
editor's "conf ≥" / "review ≥" / AI review controls, or directly in
`rules.json`. Left unset, a condition falls back to 0.65 / 0.35 / on.

The rest — the state paths, the worker counts, the detector's batch size and image
size, Ollama's timeout and context window, the review folder, the web host and
port — live in `mediasort/config/env.py`, which is the one place they are written
down. Set any of them in `.env` and it takes effect; `mediasort doctor` prints
which `.env` and which saved settings are in force.

### Ollama

Nothing is started for you: mediasort talks to the Ollama you already run, at
`MEDIASORT_OLLAMA_URL`. The Settings tab has a **Test** button that lists the
models actually installed there, which is the quickest way to confirm the URL and
pick a model — or use `mediasort doctor`.

Anything vision-capable works. `llava-llama3` is the default; `qwen2.5-vl` and
`llama3.2-vision` follow the JSON schema more reliably if you want to pull one:

```bash
ollama pull qwen2.5-vl
```

A bare name resolves to whichever tag is installed, so `llava-llama3` finds
`llava-llama3:8b` without you having to spell it out. The client asks for
structured output via Ollama's JSON-schema `format`, and falls back to extracting
the first JSON object from the reply for models that add commentary anyway.

If Ollama runs on another machine, point the URL at it — the images are sent over
HTTP, so only that host needs the GPU for the semantic pass.

### GPU

`make install` fetches the CUDA build of torch (`cu128`):

```bash
make install TORCH_VARIANT=cu130   # newer runtime
make install-cpu                   # or no GPU at all
```

The older `cu124` index publishes nothing for Python 3.13+, so on a recent
interpreter it fails to resolve rather than falling back — pick `cu128` or later.

`mediasort doctor` reports which device torch actually chose. Detection falls
back to the CPU on its own — correct, just slower.

## Rules

A rule has a **name**, a **condition** and an **action**. They are evaluated top
to bottom and **the first match wins**, so the most specific rule goes first and
the last one should match anything.

The file lives at `MEDIASORT_RULES` (`./data/rules.json`). `make install` seeds
one for you (via `make seed`); `mediasort rules init --classes cat,dog,video`
writes one from your own classes at any time — one combined rule, one per class,
one catch-all:

```jsonc
{
  "rules": [
    { "name": "cat-dog", "when": {"all_of": ["cat", "dog"]}, "action": "copy" },
    { "name": "cat",     "when": {"class": "cat"},           "action": "copy" },
    { "name": "dog",     "when": {"class": "dog"},           "action": "copy" },
    { "name": "video",   "when": {"class": "video"},         "action": "move", "folder": "video" },
    { "name": "none",    "when": {"always": true},           "action": "ignore" }
  ]
}
```

The starter is deliberately linear in the number of classes rather than one
rule per combination — five classes would otherwise mean 31 rules to read.
Add the combinations you actually want in the editor.

Videos go through the same rules as photos. YOLO samples
`MEDIASORT_DETECT_VIDEO_FRAMES` stills (8 by default) spread across each one
and sorts it by what it saw in them, so a clip of your cat lands in `Cat/`
next to the photos of it.

`video` is the one class no model looks for: it is true of a file by its
extension (`.mp4`, `.mov`, `.mkv`, ... — the full list is
`video.VIDEO_EXTENSIONS`), and every video carries it *on top of* whatever
was found in its frames. That is why it comes out of the starter as its own
`move` rule placed below the real classes rather than joining the `all_of`
combination above them: rules are tried in order, so the cat rule gets the
cat video first, and the `video` rule catches the rest — the clips nothing
was recognised in, and the ones no frame could be read from. Move it to the
top and it becomes the opposite: every video sorted as a video, whatever is
in it. That is a legitimate choice, and it is one line of the rules file.

Set `MEDIASORT_DETECT_VIDEO_FRAMES=0` to skip decoding altogether — videos
are then matched by extension alone, which is faster and blind.

A richer example — note that none of this required a code change:

```jsonc
{
  "rules": [
    // more than one cat, into a folder of its own
    { "name": "multi-cat", "when": {"class": "cat", "min_count": 2},
      "action": "copy", "folder": "Multiple-Cats" },

    // a dog but no people in frame
    { "name": "dog-alone", "when": {"all_of": ["dog", {"not": {"class": "person"}}]},
      "action": "copy" },

    // borderline confidence: put it in front of a human instead of guessing
    { "name": "maybe-pet", "when": {"any_of": [{"class": "cat", "min_confidence": 0.3},
                                               {"class": "dog", "min_confidence": 0.3}]},
      "action": "move", "folder": "Maybe" },

    // everything with no animal at all
    { "name": "no-animals", "when": {"always": true}, "action": "delete" }
  ]
}
```

**Conditions**

| Form                                                            | Meaning                                       |
| ---------------------------------------------------------------- | ---------------------------------------------- |
| `"cat"` or `{"class": "cat"}`                                    | at least one cat above the default threshold  |
| `{"class": "cat", "min_count": 2}`                               | at least two                                  |
| `{"class": "cat", "min_confidence": 0.4}`                        | this class uses its own auto-pass threshold   |
| `{"class": "cat", "min_review_confidence": 0.2}`                 | ...and its own review-band floor              |
| `{"class": "cat", "ollama_review": false}`                       | never ask Ollama about a borderline cat        |
| `{"all_of": [...]}` / `{"any_of": [...]}`                        | AND / OR                                      |
| `{"none_of": [...]}` / `{"not": {...}}`                          | NOR / NOT                                     |
| `{"any_detection": true, "min_count": 3}`                        | three or more detections of any class         |
| `{"always": true}`                                               | catch-all                                     |

`min_confidence`, `min_review_confidence` and `ollama_review` default to 0.65,
0.35 and `true` when a condition leaves them unset. When the same class appears
in more than one rule, the first rule (in priority order) that mentions it
wins whichever of these it set explicitly.

**Actions** — see [The four actions](#the-four-actions).

Whatever classes your rules mention is what the detector looks for — there is no
second list to keep in sync. `mediasort rules classes` lists everything the model
can find (80 COCO classes with the default weights).

After editing rules, you do **not** need to re-run the detector:

```bash
mediasort apply --dry-run   # what would happen
mediasort recheck           # re-decide every image from stored detections
mediasort apply             # do it
```

### The four actions

A rule says what to match and what to do about it. Each action is named for
exactly what it does to the photo, and there is no second setting that changes
what one of them means:

| action | the output folder gets | the original |
| ------ | ---------------------- | ------------ |
| `copy` | an independent duplicate, timestamps preserved | stays put |
| `move` | the photo itself | relocated |
| `delete` | nothing | to `_Trash/`, or erased |
| `ignore` | nothing | stays put; still indexed and searchable |

`folder:` on a rule chooses where `copy` and `move` put it, defaulting to the
rule's own name. Any other action name is rejected by `mediasort rules
validate`, which names the four that exist.

### The doubt rule

Some photos neither the detector nor the second opinion can settle. What happens
to those is a question every library has an answer to, so it is a **rule** —
always present, always last in the list, and not deletable:

```json
{"name": "in-doubt", "when": {"in_doubt": true}, "action": "move", "folder": "_Review"}
```

It is not in the priority race: `in_doubt` is a property of the *decision*, not
of the detections, so it can never swallow photos that an ordinary rule matched.
It only ever applies to photos flagged uncertain. Its action is one of three, and
**`delete` is deliberately not offered** — throwing away what you were unsure
about is the one thing doubt must not lead to unattended.

| action | what happens to an uncertain photo |
| ------ | ---------------------------------- |
| `move` (default) | it goes to `_Review` **instead of** its category folder. A photo nobody could settle has not earned a place under `Cat/`. |
| `copy` | it is sorted normally *and* a duplicate lands in `_Review`. |
| `ignore` | doubt changes nothing; the matched rule runs as usual. |

Because the default is `move`, a fresh install relocates anything it could not
settle into `_Review` on the first run. Uncertain photos are flagged in the
index as well, and `mediasort stats` counts them.

### Moving instead of copying

`copy` leaves your library exactly as it was and spends disk space to do it. If
you want the photo itself filed away, with nothing left behind and no second
copy, use `move`:

```yaml
- name: Cat
  match: {class: cat}
  action: move
```

Within one filesystem this is a rename: instant, and free of disk cost. It is
the right choice for a "sort this pile" folder that should end up empty, and the
only one that costs nothing on a drive with no room to duplicate a library.

One consequence follows from the original being consumed: moves are not pruned
or rebuilt on a later run the way copies are. Re-running finds the photo already
filed and leaves it alone, so re-categorising it means moving it again, not
undoing anything.

Moving is not gated: a rule whose action is `move` relocates originals on every
run. Preview with `mediasort apply --dry-run` first; it lists every source and
destination without touching anything.

### Deleting

Deletion is not gated either — a `delete` rule empties into the trash folder on
every run. `mediasort doctor` names every rule that consumes originals, which is
the one warning you get before a run does.

Running natively, that is genuinely all there is: the tool runs as you, so your
photos are as writable to it as they are to any program you start. There is no
read-only mount underneath making a mistake impossible, which is the tradeoff for
being able to point at any folder without reconfiguring anything.

Files are always *moved* to `MEDIASORT_TRASH_FOLDER`, keeping their folder
structure, so a mistake is recoverable — there is no permanent-delete mode.
Every deletion is recorded — see `mediasort history`.

Always run `mediasort apply --dry-run` first.

## Web UI

```bash
make web        # http://127.0.0.1:8765
```

Two pages, one server. The sorting page at `/` has five tabs: **Rules** (visual
editor with reordering), **Run** (the folders, the progress, and the three
steps), **Preview** (dry run and "what would these rules do to my library?"),
**Settings** (the same folders, plus Ollama URL and model, and whether to
remember this library) and **JSON** for anything the visual editor cannot
express. Each tab is addressable — `#run`, `#settings` — so a reload stays put.
Built on the standard library, no CDN assets — it works with the network off.

### Duplicates — `/duplicates`

**A separate page, over its own folders.** De-duplicating a folder and sorting a
library are different jobs, so the duplicate finder has its own input folder, its
own index, and its own scan. Pointing it at a folder does *not* add that folder
to the sorting library, does not run the detector over it, and does not put a
single row in the pipeline's counters. The reverse holds too: your sorting
library does not turn up here unless you name it here as well.

Its index follows the same rule as the sorting one — **in memory by default,
nothing left behind** — with its own opt-in toggle ("remember this scan") that
keeps the hashes in `<folder>/.mediasort/dupes.db`, separate from `index.db`.

Finds photos that only *look* the same — a re-save, a resize, a different
format — which a checksum cannot. Each photo is compared by perceptual hash and
scored as a percentage against the group's best copy:

```
group is 98.4% alike at the loosest
  100.0%  original.png   <- 200×150 PNG
  100.0%  resaved.jpg    <- the same picture at JPEG quality 60
   98.4%  small.jpg      <- the same picture at half the size
```

100% means the two *hashes* are identical, not the two files — surviving a
re-encode is the whole point of the comparison.

**Max distance** is the same measurement in the other direction: how many of the
64 hash bits are allowed to differ. 10 bits is 84.4% alike, and the page shows
that conversion next to the box. Every photo in a group is within the threshold
of the copy it is scored against — grouping finds candidates by chaining (A near
B, B near C), then splits each chain around its best copy, so a group can never
show you a pair looser than the number you asked for.

**Marking changes nothing on disk.** You mark each photo `Keep` or `Discard`; a
mark is a note to yourself, stored so a big folder can be reviewed across
several sittings without losing your place. Clicking a mark you already have
takes it back off, so "not decided yet" stays a real third state. Each group
also offers the three verdicts worth one click: **Mark best, discard rest**
(highest resolution, then largest file), **Keep all**, and **Discard all**.

Acting on the marks is a separate, explicit press: **Delete duplicates** — in
the header and again in its own card — takes everything currently marked
`Discard` and, by default, sends it to your desktop trash. Recoverable from
there until you empty it, and the button says how many photos it is about to
touch, and where they will land, before you press it. This needs `send2trash`
installed; without it, the button says so and touches nothing.

Untick **delete straight to the desktop trash** next to it and the same press
*moves* those photos into `<folder>/_Duplicates` instead, keeping the folder
structure they had — no unlink at all, undone by dragging the folder back.
Re-scanning skips that folder, so a discarded photo is not offered back as a
duplicate of the copy you kept. Either way the choice is read at the moment you
press and never stored, so the setting you can see on screen is the one that
runs.

**Not duplicates** dismisses a group instead: the grouper stops proposing that
pairing. Dismissal is per pair, so a third photo close to both can still pull
them back together — a verdict on two photos is not a verdict on a whole group.

Grouping needs a perceptual hash for every photo, so the page has two steps:
**Scan folder** hashes what is in the folder, and **Find duplicates** groups
what has been hashed. Re-scanning only re-reads files that changed.

The Run tab is laid out as the run itself: the **folders** first, since nothing
can start without them and they are the one setting a run cannot guess; then
**progress**, one bar per pass — indexed, examined by the detector, described by
the semantic pass, sorted into folders — each with its own totals and, while a
job is running, a throughput estimate measured from the last minute of real work
rather than an average over a run that changed speed; then the **steps**.

Each step owns its state and its result. The badge beside its title reads the
counters (`3,917 photos indexed`, `830 still to examine`, `nothing matched a
rule`), and the outcome of the job that step starts is printed under that step,
with how long it took — *"Finished in 0s: scanned 3899 files — nothing new or
changed since the last scan"*. A step that had nothing to do says so where you
pressed it, which is the difference between an honest no-op and a button that
looks broken. A failure is printed in the same place, in red, instead of a status
line somewhere else on the page.

**Run everything**, top right of the header, is those three steps in one press —
the web UI's `mediasort run`. It is one job on the server rather than the page
firing three in sequence, so closing the browser halfway does not leave the run
stopped where it stood; it renames itself as it advances, so the step cards and
bars below light up exactly as if each button had been pressed in turn, and the
result names all three steps.

A **Where you are** panel above them reads the same counters and says in one
sentence which button comes next — including the cases that otherwise look like a
broken button: photos still waiting for the detector, and photos the detector
examined without finding anything the rules ask for. Preview explains an empty
result rather than showing empty tables.

Saving in Settings applies immediately: the server re-reads every layer and
rebuilds itself, no restart. It refuses while a job is running, since a stage
holding settings that just changed underneath it would be working from two
configurations at once. It writes the same `data/settings.json` that
`mediasort config set` writes, so the two front ends cannot disagree — and until
photo folders are set, the Run tab says so instead of letting you scan nothing.

It is unauthenticated by design — it edits your settings and can delete your
files. Keep `MEDIASORT_WEB_HOST` on `127.0.0.1`.

## Commands

Every option, plus recipes and troubleshooting, is in
[docs/CLI.md](docs/CLI.md). The summary:

```
mediasort scan            index new / changed images (safe to re-run any time)
mediasort run             full pipeline: scan, all three passes, apply
mediasort detect          detection pass only              [--limit N]
mediasort adjudicate      second opinion on the unsure ones [--limit N]
mediasort analyze         semantic pass only               [--limit N]
mediasort apply           execute the rule actions   [--dry-run]
mediasort recheck         re-decide from stored evidence (no GPU, no Ollama)

mediasort config show     the folders and Ollama settings, and their source
mediasort config set      save them   [--input F] [--output F] [--ollama-url U]
mediasort config unset    hand a setting back to .env or the default

mediasort rules show      the active rules, in evaluation order
mediasort rules init      write a starter file        [--classes cat,dog,bird]
mediasort rules validate  check it parses         [--check-classes]
mediasort rules actions   what a rule can do
mediasort rules classes   what the detector can find
mediasort web             rules editor + control panel     [--port N]

mediasort stats           per-stage progress and category counts
mediasort doctor          GPU, weights, rules, Ollama connectivity
mediasort verify          confirm originals and links still exist
mediasort duplicates      files sharing a SHA256 hash
mediasort history         originals that were moved or removed
mediasort search TERM     free-text search over the Ollama metadata

mediasort retry           requeue failed  [--stage detect|adjudicate|analyze]
mediasort reset STAGE     reprocess      detect | adjudicate | analyze | all
mediasort clean           --links | --missing | --vacuum
mediasort export          dump the index               [--format json|csv]
```

There is no separate `resume` command because there is no separate resume
path — every command is resumable by construction. Re-running `run` after a
`Ctrl-C` continues from the database.

Ctrl-C once finishes the batch in flight and exits cleanly; twice forces it.

## How it works

**Scanner.** Walks the input folders and compares each file's size and mtime
against the database. It prunes whatever the output and trash folders resolve to,
since both default to somewhere inside the library — otherwise a `copy` run would
index its own copies, and a `delete` run would find the trash on the next pass. A match is skipped without opening the file, which is
what keeps a re-scan of a 300k-photo library to a couple of minutes. Only new
or changed files get a SHA256 and an image-header probe (dimensions, format,
EXIF capture date). A changed file resets its own pipeline state and discards
its stale detections. Files that vanish are flagged `missing`, not deleted, so
an unplugged drive never destroys work.

**Detector.** One model instance doing batched inference, fed by a thread pool that
decodes the next batch while the GPU works on the current one. Detections are
stored down to the lowest `min_review_confidence` any rule's class band asks
for, not the (possibly higher) `min_confidence` that band auto-passes at — the
extra rows are what make the review flag and later threshold changes possible
without re-running the model.

A video goes through the same model by a second route: eight stills sampled
across it (`MEDIASORT_DETECT_VIDEO_FRAMES`) run as one batch, so a video costs
what eight photos cost. The frames then have to agree on one answer, because
the index stores detections per *file* — per class, the frame that saw the most
of it wins, ties going to the more confident frame. That keeps counts meaning
something: a `min_count: 2` rule still needs one frame that actually held two.
The boxes stored for a video are that winning frame's, in its coordinates.
Every video also carries the `video` class on top of whatever was found, which
is what makes a `{"class": "video"}` rule the fallback under the real ones —
see [Rules](#rules).

**Second opinion.** The detector is cheap and sure of itself; the vision model
is slow and can actually look. So the expensive one is asked exactly one
question, about exactly the images the cheap one could not answer: a class
whose band turns `ollama_review` on, and whose detection falls between that
class's `min_review_confidence` and `min_confidence` — the same band that
raises `needs_review` — is put to Ollama as *is this really there?*

The answer is `present`, `absent` or `unsure`, and **`unsure` is a real answer**:
the prompt says so, and a reply that cannot be parsed is read as `unsure` too.
Guessing `absent` from a broken reply would quietly discard a photo; guessing
`present` would sort it wrongly. Neither is worth the risk of not asking a
person.

What a verdict does is deliberately small. `present` promotes that class's best
borderline box to the decision threshold — the *best* one, because a verdict
establishes that the subject is there, not how many there are, so a
`min_count: 2` rule still needs the detector to have been sure twice. `absent`
drops the class. `unsure` changes nothing, which leaves the photo exactly as
uncertain as it was, which is what `_Review` is for. A box already above the
threshold is never touched: the stage settles doubt, it does not overrule
certainty.

Then the ordinary rules engine runs. It cannot tell an adjudicated detection
from a confident one, and that is the point — the category, the action and the
review flag all fall out of your rules the same way they always did, and the
decision engine stays AI-free.

Verdicts are stored in their own table rather than folded into `detections`, so
the detector's raw numbers survive intact and `recheck` can replay both in
order. Escalation is therefore paid for once: edit your rules afterwards and the
verdicts are reused, not re-asked.

The queue is the review flag itself, which keeps the two honestly in sync —
there is no second definition of "uncertain" to drift. Turn `ollama_review` off
for a class, or have Ollama unreachable, and an uncertain photo of that class
goes straight to `_Review`: exactly what the tool did before this existed.

**Rules engine.** Pure logic, zero AI (`domain/`). Conditions are
a composable tree — a new operator is a new `Condition` subclass plus one
registry entry. The first matching rule supplies both the category and the
action. `needs_review` stays independent of the rules: it reports uncertainty
that survived both models, which is a property of the image, not of your policy.

**Actions.** Each action is a strategy object in a registry (`actions/`),
planned before anything is written and executed afterwards. That split is what
makes `--dry-run` an honest preview rather than a second code path, and adding
`move`, `tag` or `upload` means writing one class and registering it — the
rules engine, CLI and web UI pick it up with no changes.

**Analysis.** Runs only on images the rules did not `ignore`, which is the single
biggest time saver in the design — a typical library skips 80-90% of the work
here. Images are downscaled to 1024px before encoding. The detector's findings
are passed as a prior in the prompt.

**Applying.** Computes the entire desired tree from the database first, prunes
links that no longer belong, then creates what is missing. That ordering makes
`sort` idempotent: running it twice creates nothing the second time. Every link
is recorded in the `links` table, so pruning removes exactly what this tool
created and nothing else.

### Where the index lives — and whether it exists

**By default it does not.** The index is held in memory for the run and dropped
when the command ends. Point the tool at a folder, sort it, walk away: nothing
is written into your photos, and there is no state anywhere to go stale, get
confused between libraries, or need cleaning up.

That default is right because the index is only ever a *cache of expensive
work* — what YOLO saw, what Ollama said. It pays for itself when the same
folder is worked on twice, and costs you a stray file when it isn't.

For a library big enough that losing a half-finished detection run would hurt,
turn it on:

```bash
mediasort config set --save-index      # or the Settings tab's "Remember this library"
mediasort config set --no-save-index   # back to leaving nothing behind
```

The index then lives **inside the library it describes**, at
`<photo folder>/.mediasort/index.db` — not in the app folder. One index per
library, which has three consequences worth knowing:

- Pointing the tool at a different folder cannot disturb this one's work.
- The saved work travels with the drive. Plug it into another machine and the
  detection results are still there.
- The `.mediasort` folder is hidden, and the scanner skips dotted directories,
  so it never indexes its own index.

Setting `MEDIASORT_DB` to a specific file overrides both — naming a location is
itself a request to save one.

An index written by an older version of this tool is **silently thrown away and
rebuilt**, never refused. Everything in it is re-derivable by re-scanning, so
there is no version to bump, nothing to migrate, and no error to read before you
can get on with sorting photos.

### Schema

```
images         id, path, filename, root, hash, phash, size, mtime, width, height,
               format, taken_at, category, action, needs_review, detect_state,
               adjudicate_state, analyze_state, missing, deleted, error
detections     image_id, class, confidence, x1, y1, x2, y2, model
adjudications  image_id, class, verdict, confidence, model
metadata       image_id, json, model
links          image_id, link_path
action_log     image_id, action, detail, created_at
dupe_dismissals  image_id_a, image_id_b
```

`category` is the name of the rule that matched and `action` is what it asked
for. Because raw detections are stored down to the *review* threshold rather
than the decision threshold, and verdicts are kept beside them rather than
overwriting them, changing rules or thresholds only needs `mediasort recheck` —
neither model runs twice.

`detect_state` / `adjudicate_state` / `analyze_state` are `0 pending, 1 running,
2 done, 3 error, 4 skipped` rather than plain done-flags — a single column has
to express "crashed halfway" and "deliberately not applicable" for resume to be
correct. `4 skipped` is what makes the cascade readable: a photo the detector
was sure about is `skipped` for adjudication, not pending, so the description
pass can tell "settled" from "still waiting on a verdict" and never writes off a
photo the second opinion is about to rescue from `ignore`.

Stages claim rows with an atomic `UPDATE`, so they never collide on the same
image. Run one mediasort instance at a time per database: each stage resets rows
left in `running` at startup, which assumes any previous run is dead rather than
concurrent.

Query it directly whenever you like:

```sql
-- every Maine Coon, no re-scanning
SELECT i.path FROM images i JOIN metadata m ON m.image_id = i.id
WHERE m.json LIKE '%Maine Coon%';

-- photos with two or more dogs
SELECT i.path FROM images i JOIN metadata m ON m.image_id = i.id
WHERE json_extract(m.json, '$.dog_count') >= 2;
```

### Layout

The packages are layered; each may import only from the ones above it.

```
errors, imaging, filesystem   cross-cutting basics
domain/       rules, decisions, adjudication, value types — pure, no I/O, no
              dependencies
config/       what you chose: grouped by concern (sections), assembled from the
              layers below (env) — .env (dotenv) and the overlay the UI and
              `config set` write (overrides, which also owns the one-run
              `--input` / `--output`). The only package that touches os.environ.
storage/      the SQLite index: one engine, one repository per table
scanning/     the filesystem, indexed incrementally
detecting/    adapters over the two inference engines, behind pipeline.ports.
analyzing/    One Ollama client serves both questions it is asked: describe
              this, and is this really there.
actions/      what happens to a photo: base, builtin, planner, executor
pipeline/     the stage loops and the runner — no idea what YOLO or Ollama are
services/     one function per use case; returns plain data
interfaces/   cli/ and web/ — parsing and presentation only
```

The application makes no assumption about where it runs. Every path is a plain
path on this machine, and stays one: the output tree is made of real files, so
there is nothing in it whose *contents* another program has to be able to
resolve, and no path-rewriting layer to configure.

## Extending it

The database is deliberately the interface, so most additions are a new stage
plus a new state column rather than a rewrite:

- **more animals, different logic** — rules only, no code change
- **a new kind of action** — subclass `Action`, register it, done
- **a new condition operator** — subclass `Condition`, `register_condition(...)`
- **blur detection / best-photo picking** — a new column, computed during the scan
- **embeddings + natural-language search** — a new table, a new stage that
  claims rows the way `claim_analyze` does
- **EXIF / GPS indexing** — the scanner already opens the EXIF block
- **face recognition for pets** — a stage after YOLO, reusing the stored boxes

## Tests

```bash
make test                       # .venv/bin/pytest -q tests
```

Tests covering the rules engine and its parser, the decision engine, the
scanner's incremental behaviour, action planning and execution (including every
deletion lock), database claiming and migration, the Ollama client against a
fake server, both stage loops against fake inference engines, frame sampling
against real videos written on the fly, the web operations, the configuration
layers, and a CLI smoke test per command. They use only temporary directories —
no GPU, no Ollama, no real photo library, and neither your `.env` nor your
saved settings can reach them.

## Notes and limits

- `llava-llama3` fills in counts, breeds and environment reliably but often
  returns an empty `activities` list. If that field matters to you, a newer
  vision model is the fix — nothing in the pipeline changes.
- HEIC/HEIF works out of the box — `pillow-heif` is a hard dependency and the
  extensions are indexed by default. RAW formats are not handled.
- Videos are sorted by their content: YOLO runs on
  `MEDIASORT_DETECT_VIDEO_FRAMES` stills sampled across each file, and the
  frame that saw the most of a class speaks for the whole video (so
  `min_count` still means something). Every video also carries the `video`
  class, which is what a `{"class": "video"}` rule below the real ones
  catches. Two things to know: a subject that appears between two sampled
  frames is missed — raise the count for long clips — and the box
  coordinates stored for a video belong to one frame, not to the file.
- Ollama sees a video as a single frame from halfway in, both for the
  description pass and for a second opinion. That is a real limitation for
  anything that only makes sense in motion.
- `move` within one filesystem is a rename and costs nothing; across filesystems
  it is a copy followed by a delete, and takes as long as the bytes do.
- `hundreds of thousands of images` is the design target: SQLite with WAL
  handles it comfortably, and the sorter holds one entry per *sorted* image in
  memory (only the pet photos, not the whole library).
- The database lives at `MEDIASORT_DB` (`./data` by default). Back that up rather
  than the output tree — the output tree can always be rebuilt with
  `mediasort apply`.
- Run one mediasort at a time per database. The web UI runs one job at a time by
  construction; a CLI command started alongside it is not coordinated with it.
