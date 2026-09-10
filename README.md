# siftr

Find media by example.

Off-the-shelf taggers know about "dog", "beach" and "car". They do not know about
the specific things in *your* library — a particular type of ceramic glaze, your
grandmother's handwriting, one model of vintage amplifier, your friends' faces.
siftr lets you teach it those: drop a handful of examples in a folder, and it
finds everything else in your library that matches.

It works on images and videos, runs entirely on your own machine, and never
moves your originals unless you tell it to.

## How it works

siftr embeds every file in your library into a CLIP vector space once, then
answers queries against those stored vectors:

- **Teaching a concept** averages your examples' embeddings into a *prototype*,
  and a file matches when its cosine similarity to that prototype clears a
  threshold. With a handful of examples and no negatives, a prototype beats
  anything with more parameters — there is not enough data to fit them.
- **People** use a separate face pipeline (RetinaFace detection, ArcFace
  embeddings). Every reference photo you give is kept rather than averaged, and
  matching takes the best similarity across them, so one person can look
  different at different ages and angles.
- **Videos** are sampled — a handful of frames spread through the clip rather
  than every frame. A concept only has to appear somewhere in a video for the
  file to match, and full-decode embedding costs orders of magnitude more for
  almost no extra recall.

## Install

```bash
pip install -e .
```

Face recognition and faster video decoding are optional extras:

```bash
pip install -e '.[faces,video]'
```

Without `video`, siftr shells out to `ffmpeg` for video frames. Without `faces`,
everything works except people.

## Use

Index a library. This is the slow part; re-runs skip unchanged files.

```bash
siftr index ~/Pictures
```

Teach a concept from a folder of examples:

```bash
siftr teach ceramic-glaze ~/examples/glaze --negatives ~/examples/not-glaze
```

`--negatives` is optional but worth supplying. With counter-examples siftr fits
the threshold to what actually separates your concept from the rest of your
library; without them it falls back to a conservative default.

Search:

```bash
siftr search --concept ceramic-glaze --scores
```

Or skip teaching entirely and search straight from a folder of examples:

```bash
siftr search --examples ~/examples/glaze -n 40
```

Or search by text, with no examples at all:

```bash
siftr search --text "a hand-thrown bowl with a crackled finish"
```

Collect results into a folder. The default is symlinks, so your originals stay
where they are:

```bash
siftr search --concept ceramic-glaze --output ~/sorted/glaze
siftr search --concept ceramic-glaze --output ~/sorted/glaze --mode copy
```

### People

Register someone from a folder of photos of them:

```bash
siftr add-person "Nadia" ~/refs/nadia --rematch
```

siftr uses the largest face in each reference photo, so group shots where they
are in front still work. `--rematch` re-checks faces siftr has already found in
your library, so you do not need to re-index after adding someone.

```bash
siftr search --person "Nadia"
```

### Everything else

```bash
siftr status              # index statistics
siftr concepts            # what you have taught
siftr people              # who is registered
siftr apply <concept>     # store tags for a concept across the index
siftr rematch             # re-check unidentified faces against all people
siftr forget concept <name>
siftr forget person <name>
```

## Tuning

`siftr teach` reports a **cohesion** score — how similar your examples are to
each other. Below about 0.70 the concept is vague, and the fix is almost always
to *narrow* the example folder rather than add to it. Ten tightly-related
examples beat fifty loose ones.

If a concept is too loose or too strict, override its threshold per search
rather than re-teaching:

```bash
siftr search --concept ceramic-glaze --threshold 0.62
```

Note that text and example searches are not on the same scale — CLIP text
similarities run much lower than image-image ones — so a threshold tuned for
`--examples` will reject everything under `--text`.

## Desktop app

A photo-slap-style UI lives in `desktop/`. Tags are taught by dragging photos
onto them, a photo can carry several tags at once, and matched tags are written
into filenames.

```bash
pip install -e '.[ui,faces]'      # the API the UI talks to
cd desktop && npm install && npm run dev
```

To build a real macOS app:

```bash
cd desktop && npm run dist        # -> release/0.1.0/siftr-<version>-arm64.dmg
```

The build is unsigned, so Gatekeeper wants a right-click → Open the first time.
It bundles the UI but **not** Python: torch alone is ~590 MB, which would turn a
117 MB DMG into roughly 1.5 GB before a single model is downloaded. The app finds
`siftr` on `PATH` instead. A packaged app does not inherit your shell `PATH`, so
if siftr lives in a virtualenv, point at it directly:

```bash
SIFTR_BIN=/path/to/.venv/bin/siftr open -a siftr
```

If it cannot find the service, the app says so and tells you this, rather than
failing silently.

The app starts `siftr serve` itself and talks to it over a local HTTP API. That
indirection exists because loading CLIP costs several seconds; a resident process
pays it once instead of on every click.

- **Drag photos onto a tag** to teach it. Dropping onto the target at the bottom
  of the rail makes a new tag. Photos can come from the grid or from Finder.
- **Hold ⌥ while dropping** to teach a counter-example instead — "this is *not*
  that". Counter-examples only tighten a tag; they never loosen it.
- **Drop photos of someone** onto the People section to name them, and siftr
  finds them everywhere else. Faces it has seen repeatedly but cannot name are
  offered under **Unnamed faces** — click one to name the whole group at once.
- **Select photos and use "add to tag"** to pin a tag on many at once.
- **Click tags** in the rail to filter, and use **ANY / ALL** to switch between
  the union and the intersection of what you selected.
- **Click a chip** on a photo to say the model got that one wrong. The correction
  survives re-scoring, and the photo becomes a training example for that tag.
- **Filenames are updated automatically** as the library is scored. `Undo
  renames` reverses the last batch; when filenames and tags disagree afterwards,
  a banner says how many and offers to re-apply.
- **Long jobs can be cancelled** from the progress bar; work already done is
  kept.

### About the automatic renaming

Every rename batch is journaled to `.siftr-renames.json` at the library root, so
`Undo renames` can always walk it back. Only bracket groups matching a known
siftr tag are touched — if mediate has written `[2]` or `[site 3]` into a
filename, those stay exactly where they are. Re-scoring converges rather than
appending, and a tag that stops matching loses its bracket. Nothing is ever
overwritten: a name that is already taken gets a numeric suffix.

## Where things live

The index is a single SQLite file at `~/.siftr/index.db` (override with
`--db`, or `SIFTR_HOME`). Nothing about your library is uploaded anywhere; the
only network access is the one-time model download.

| Module | Role |
|---|---|
| `cli.py` | argparse CLI; imports torch lazily so metadata commands stay fast |
| `db.py` | SQLite schema and queries; embeddings as float32 BLOBs |
| `embed.py` | CLIP image and text embeddings via open_clip |
| `concepts.py` | few-shot prototype learning and threshold selection |
| `faces.py` | InsightFace detection/recognition and person matching |
| `media.py` | file discovery, EXIF-correct loading, video frame sampling |
| `index.py` | the indexing pass, concept application, face re-matching |
| `search.py` | ranking the index by concept, examples, text, or person |
| `organize.py` | materializing results as symlinks, copies, or moves |
| `vectors.py` | normalization, serialization, cosine, centroid |

## Tests

```bash
pytest
```

The suite is hermetic — it never downloads CLIP or InsightFace. A deterministic
fake embedder and a stub face analyzer stand in, because the logic worth testing
(storage, thresholds, reductions, placement) does not depend on which model
produced the vectors.
