# siftr — Architecture Notes

Few-shot media search. Teach a concept from a folder of example images, then find
everything matching it across a local image/video library. Also tags known people
from reference photos. Python ≥3.10, CLIP embeddings via `open_clip`, faces via
InsightFace, index in one SQLite file.

Replaces an earlier `media-org` prototype whose classifier was a
randomly-initialized, never-trained linear head on ResNet50 — every score it
produced was noise. Nothing from that codebase survives except the general idea.

## Module map (`siftr/`)

| Module | Role |
|---|---|
| `cli.py` | argparse; lazy imports so metadata commands skip the torch import |
| `db.py` | SQLite schema + queries; embeddings as float32 BLOBs |
| `embed.py` | `Embedder`: CLIP image/text embeddings, lazily loaded model |
| `concepts.py` | prototype learning and threshold selection from example folders |
| `faces.py` | InsightFace detection/ArcFace embeddings; person matching |
| `media.py` | discovery, EXIF-correct loading, video frame sampling |
| `index.py` | the threaded indexing pipeline, `apply_concept`, `rematch_faces` |
| `search.py` | ranking by concept / examples / text / person |
| `organize.py` | symlink/copy/move results into folders |
| `regions.py` | person crops from face boxes — measured as unhelpful, off by default |
| `vectors.py` | normalize, blob (de)serialize, cosine, centroid |

## Design decisions worth keeping

- **Every stored embedding is L2-normalized** (`vectors.py`), so a dot product
  *is* cosine similarity. The invariant lives in one module; nothing else
  handles magnitudes.
- **Prototype, not a trained head.** A concept is the normalized mean of its
  examples. With 5–50 examples and no negatives there is not enough data to fit
  anything larger, and a prototype cannot overfit or diverge.
- **Calibration must exclude the taught examples** (`service._library_scores`).
  Examples are usually *in* the library and necessarily score highest against a
  prototype built from them, so leaving them in the negative pool makes the
  widest gap "my examples versus everything else" — the threshold lands above
  every real match and the tag finds only the files it was taught from. Measured
  on a 27-photo library: 0.786 and 0/8 recall with them in, 0.600 and 5/8 with
  them out. It also made the result unstable, swinging between 0/8 and 5/8 on
  different five-example subsets of the same concept.
- **Threshold selection is the part that actually matters** (`concepts.learn`):
  with negatives, the cutoff goes just above the strongest negative but is
  clamped to the `percentile`-th weakest positive — without that clamp, one
  negative that genuinely resembles the concept pushes the threshold above every
  positive and the concept silently matches nothing. Positives-only falls back to
  the positives' own percentile, floored at `MIN_THRESHOLD` (0.60).
- **Per-file score is the max over frames** (`concepts.best_per_file`). A concept
  in one sampled frame makes the whole video a match; averaging would bury it.
  This is also what collapses a video's frames into a single search hit.
- **Videos are sampled, not fully decoded**, and sample points are inset 5% from
  both ends (`media._sample_times`) — t=0 is very often a black frame and the tail
  is often a fade or past the last decodable packet.
- **Faces keep every reference per person, never a centroid** (`db.person_faces`,
  `faces.match`): matching takes the best similarity across a person's
  references, so one person can look different at different ages and angles. A
  single averaged vector measurably loses accuracy here.
- **`file_faces.person_id` is `ON DELETE SET NULL`**, so forgetting a person
  keeps the expensive detection work; `rematch` can then re-assign it.
- **Symlink is the default `--output` mode.** These are the user's originals; a
  tagger that reorganizes them by default eventually loses something.

## Duplicate detection

**CLIP is the wrong tool for this and was not used.** It is trained to be
invariant to exactly what separates a duplicate from a similar photo. Measured on
one photo plus resize/re-encode/brightness/crop variants against genuinely
different shots of the same scene, CLIP's cosine separated the two classes by
**0.003**. A perceptual hash separated them cleanly. They answer different
questions — "same subject" versus "same photograph".

- **Hash size is not free.** At 64 bits the same fixture gave *no* separation
  (duplicates 0-2 bits differing, different photos 2-6). At 256 bits
  (`DCT_SIZE=64`, `KEEP=16`) duplicates landed at 0.8-14.9% of bits and different
  photos at 18.0-23.5%. `DEFAULT_DISTANCE = 0.12` sits in that gap, below the
  hardest duplicate (a crop, 14.9%) — raising it toward 0.16 catches crops at
  some cost in precision.
- **Fixtures must have texture.** A DCT hash keys on high-frequency content; flat
  synthetic gradients are pathological and make any threshold look broken. The
  test fixtures add noise for this reason.
- **Exact duplicates are found separately**, by BLAKE2b over the file bytes.
  Someone may delete based on this, and a hash collision story is not worth
  telling when certainty is free.
- **The keeper heuristic is resolution, then bytes, then the name.** A compressed
  12 MP frame beats a lossless 800px export of it. When copies are identical only
  the name can decide, so names advertising themselves as copies ("copy", "(1)",
  "-2") lose before modification time is consulted — mtime sounds authoritative
  but some copy tools preserve it and others reset it.
- **All-pairs Hamming, chunked.** ~47s over 50k x 256-bit hashes with
  `np.bitwise_count`, peak temporary a few hundred MB. An LSH or BK-tree index is
  what this wants an order of magnitude beyond that.
- Hashes are computed on the indexing worker threads, where they are free next to
  the model work, and stored on `files` (schema v6).

## What tagging is good and bad at

Measured on 27 real photographs (13 with visible arm tattoos, 14 people without),
teaching "tattoo-sleeve" from five examples and scoring the rest:

| Training | Threshold | Recall | False positives |
|---|---|---|---|
| 5 positives | 0.600 | 5/8 | 0/14 |
| 5 positives + 5 negatives | 0.533 | **7/8** | 0/9 |
| 9 positives | 0.600 | 3/4 | 0/14 |

- **Counter-examples are worth far more than more examples** for a subtle
  attribute: five negatives took recall from 5/8 to 7/8, while nearly doubling
  the positives did not help. This is the case ⌥-drop exists for.
- **Precision is the easy half.** Zero false positives in every configuration.
  Recall is where these concepts lose.
- **What it learns may not be what was asked.** The tag found tattoo *parlour*
  scenes where the tattoos are small in frame, and missed a field portrait and an
  extreme close-up of a tattoo. It had learned something closer to "tattoo
  culture scene" than "tattooed arm" — unsurprising, since CLIP embeds the whole
  frame and the examples shared a setting.
- **Whole-image concepts are much easier.** Bicycles/sunsets/blueprints separated
  by ~0.4 cosine with 7/7 recall and no false positives; an attribute occupying a
  few percent of the frame separates by ~0.02-0.05. Measured directly: a feature
  covering 1% of the frame moves the embedding by 0.012 and yields +0.008
  prototype separation; at 50% it is +0.093.
- **Do not probe concepts with CLIP text-text similarity.** "tattoo sleeves" vs
  "bare untattooed arms" scores 0.802, *higher* than vs "heavily tattooed"
  (0.787) — shared vocabulary dominates. It says nothing about image behaviour.

### Person crops were tried and did not work

The obvious fix for the above is to embed **person crops** rather than whole
frames: a sleeve that is 3% of a photo is a large share of a crop. It is built
(`regions.py`, `siftr index --person-crops`) and it is **off by default, because
measurement says it is worse**.

Ranking held-out positives above negatives on the same 26 photographs:

| Representation | mean pos | mean neg | AUC |
|---|---|---|---|
| whole frame | 0.614 | 0.406 | **0.90** |
| crop 3.2w x 4.5h | 0.623 | 0.477 | 0.82 |
| crop 5.0w x 5.0h | 0.625 | 0.433 | 0.80 |
| crop 6.5w x 6.0h | 0.585 | 0.416 | 0.80 |
| crop 4.0w x 7.0h | 0.629 | 0.456 | 0.82 |
| crop 8.0w x 8.0h | 0.585 | 0.407 | 0.80 |

End-to-end recall was identical (5/8, and 7/8 with negatives) with crops on or
off, because the prototype is still built from whole-frame examples. Building the
prototype from crops too made it *worse*, not better.

Two reasons, neither reachable by tuning the box. CLIP is trained on whole images
paired with captions, so a tight crop is out of distribution. And per-file score
is a max over that file's embeddings, so extra vectors raise the **negative**
ceiling as much as the positive one — mean negative rose 0.406 -> 0.477.

Kept rather than reverted because this is one attribute on one dataset, and an
attribute centred on a person (a hat) may behave unlike one spread across them.
Measure before turning it on.

### A bigger or different embedding model does not help either

Benchmarked on the same 26 photographs, averaging AUC over several choices of
five training examples:

| Model | dim | AUC |
|---|---|---|
| **ViT-B-32 laion2b** (current) | 512 | **0.944** |
| ViT-B-16 laion2b | 512 | 0.937 |
| ViT-L-14 laion2b | 768 | 0.927 |
| ViT-B-16-SigLIP2-256 | 768 | 0.910 |
| ViT-B-16-SigLIP-384 | 768 | 0.944 |
| ViT-SO400M-14-SigLIP-384 | 1152 | 0.946 |

Nothing beats the small fast model by more than noise, including one seven times
its size. **Do not swap the embedding model hoping to improve attribute tagging.**

### What does work: open-vocabulary detection, at a price

OWLv2 (`google/owlv2-base-patch16-ensemble`) queried with "a tattooed arm"
ranked the same set at **AUC 1.000** — every positive above every negative,
positives averaging 0.392 against 0.161.

It cannot replace CLIP as the index representation:

- **99x slower**: 0.59 images/s against 58.9. A 50,000-photo library is 23 hours
  against 14 minutes.
- **Per query, not once.** Embeddings are computed once and answer any later
  question; detection re-runs the whole library for every new query.
- It takes a **text query**, not examples, so it does not fit teach-by-dropping.

The shape that would work is two-stage: CLIP narrows the library to a few hundred
candidates from the existing index, then OWLv2 verifies those. 200 candidates is
about six minutes, which is tolerable for a deliberate query, and it buys
near-perfect precision on the result. Not built — it adds `transformers` and a
~600 MB model for an optional capability.

## Indexing performance

`build_index` runs decode, CLIP preprocessing and face detection on a thread
pool, and the model forward pass on the calling thread in batches. The split
follows the measurements, which were not what they looked like from the outside:

| Stage | Serial | Parallel/batched | Gain |
|---|---|---|---|
| HEIC decode (12 MP) | 21.5/s | 4 threads | 2.5x |
| CLIP preprocess | 20.0/s | 4 threads | 4.2x |
| CLIP forward | 93.7/s | batch of 24 | 2.4x |
| Face detect | 15.3/s | 4 threads | 3.6x |

- **Preprocessing, not the model, is ~92% of the "embedding" cost.** Resize and
  normalize on CPU dwarf the forward pass. That is why `Embedder` exposes
  `preprocess` and `embed_tensors` separately: the expensive half threads, the
  cheap half batches.
- **Everything except the forward pass releases the GIL** — image codecs,
  torchvision transforms, and ONNX Runtime — which is what makes threads worth
  anything here. Measure before assuming otherwise.
- **Measure MPS with `torch.mps.synchronize()`.** It is asynchronous; an
  unsynchronized forward looked like 8364 img/s and was actually 223.
- **Workers carry tensors, not images.** A preprocessed 224x224 tensor is
  ~600 KB where the 12 MP photo is ~36 MB, which is what makes the in-flight
  window (`pool_size * 4`) affordable.
- **Only the calling thread touches SQLite.** Connections are not safe to share,
  so `write()` is called from the consumer loop alone.
- **Commits are every 50 files, not per file.** An fsync per photo dominates once
  decoding is no longer the bottleneck, and a batch still bounds what an
  interrupted scan loses.
- **`FaceRecognitionUnavailable` must not cost a file its embeddings.** The
  worker returns its tensors with `faces_unavailable` set; the consumer announces
  once and stops asking for faces. An early version `continue`d instead and
  silently dropped every in-flight file.

Measured end to end on 60 x 12 MP HEIC with faces: 6.21/s single-threaded to
25.55/s on eight workers — **4.1x**, or 2.24 hours down to 0.54 for a
50,000-photo library.

## Organize modes and companions

`service.organize` applies the library's stored policy (`organize_mode` in
`meta`): `rename` writes tags into filenames, `move` files matches into each
tag's `destination`, `off` leaves the filesystem alone. Both write to the same
rename manifest, so one undo covers either.

- **Everything acts on companion groups, never lone files** (`companions.py`).
  A Live Photo is a still plus a motion clip paired *by stem*; siftr scores each
  independently, so the two halves match different tags and would be given
  different names. Every member takes the **primary's** tags (the still wins over
  the clip), which keeps the stems identical on both sides of a rename or move.
  Using the primary's tags rather than the union is deliberate: the union would
  put tags on the still that only the clip matched, and re-scoring one half would
  then change the other's name.
- **Grouping is by (directory, casefolded stem).** ExifTool's
  `ContentIdentifier` would confirm a true pair, but grouping two unrelated
  same-stem files is harmless — they simply keep matching stems — whereas failing
  to group a real pair breaks it. Group by default.
- **Companions that were never indexed still travel.** `find_companions` reads
  the directory, so an `.AAE` siftr does not embed is still moved with its photo.
- **A bucket of only sidecars is skipped** rather than having a primary invented
  for it.
- **Move mode: highest-scoring claiming tag wins**, ties broken alphabetically so
  the result does not depend on dict ordering. Contested files are reported, not
  silently filed.
- **`shutil.move` on EXDEV**: a destination on another volume cannot be reached
  by `os.rename`.
- **`save_concept` must not clear `destination`** — re-teaching a tag replaces
  its prototype, not where its matches are filed.

## Gotchas

- **`PRAGMA foreign_keys = ON` is required** (`db.Database.__init__`). SQLite
  defaults it off, which would make every `ON DELETE CASCADE` in the schema
  silently do nothing. `test_deleting_file_cascades_to_embeddings` guards it.
- **`upsert_file` clears the file's embeddings, faces and concept rows.**
  Re-indexing a changed file must not leave stale vectors beside the new ones.
- **`is_unchanged` requires an embedding row to exist**, not just a `files` row —
  otherwise a scan interrupted between the two would treat the file as done and
  never embed it.
- **Global flags work on both sides of the subcommand** (`cli._global_flags`,
  `cli.resolve_globals`). argparse's subparser action parses into a *fresh*
  namespace and copies its attributes onto the parent's, so a normal default on
  either parser clobbers a value given on the other. Both sides use
  `default=argparse.SUPPRESS` and the real defaults are applied in
  `resolve_globals`. `test_subcommand_does_not_clobber_a_flag_given_earlier`
  guards this.
- **`ImageOps.exif_transpose` on load** (`media.load_image`). A portrait phone
  photo embeds sideways without it, degrading both detection and similarity.
- **Package directories are pruned** (`media._is_package`): walking into
  `Photos Library.photoslibrary` would index thousands of internal derivatives
  alongside the real masters.
- **Dimension mismatches must say what to do.** Switching CLIP models invalidates
  every stored vector; `index.apply_concept` and `search._search_vector` raise a
  message naming the fix rather than a raw broadcast error.
- **Text and image similarities are not on the same scale.** CLIP text scores run
  far lower, so `search.by_text` defaults to no threshold — rank, don't cut.
- **Blind `except` around third-party decoders is deliberate.** PIL, ffmpeg, ONNX
  and CLIP raise a wide, unstable range of types, and one unreadable file must
  not abort a multi-hour scan. Failures land in `IndexStats.errors`. `BLE001`
  and `S112` are therefore not in the ruff selection.
- **`zip(..., strict=True)` everywhere vectors pair with rows.** A length
  mismatch there would silently misalign embeddings against their files.
- **Indexing commits per file**, so an interrupted scan keeps its work.

## Desktop app (`desktop/`)

Electron 43 + React 19 + TS + Vite 8, deliberately mirroring photo-slap's
structure and look. Main spawns `siftr serve` and owns the API token; the
renderer never holds it.

| Piece | Role |
|---|---|
| `electron/main.ts` | window, CSP, navigation guards, `siftr://` protocol, IPC proxy |
| `electron/pythonService.ts` | spawns `siftr serve`, reads the token off stdout |
| `electron/preload.ts` | the entire renderer surface via `contextBridge` |
| `src/App.tsx` | state, job polling, teach/score/undo orchestration |
| `src/components/TagRail.tsx` | tags as drop targets, ANY/ALL toggle |
| `src/components/Grid.tsx` | windowed thumbnail grid, drag source, tag chips |
| `electron/menu.ts` | application menu; `menuTemplate()` is pure data so it can be tested |
| `build/make-icon.py` | draws `icon.png` + `icon.icns` — run it to change the mark |
| `src/filter.ts` | pure ANY/ALL filtering, tested without rendering |

### Desktop gotchas

- **The preload MUST build as CommonJS.** A sandboxed Electron preload cannot be
  ESM. Built as ESM it fails with only "Unable to load preload script" in the
  console, `window.api` is undefined, and the renderer dies on first access. Use
  vite-plugin-electron's `simple({ main, preload })` form — its dedicated preload
  path emits CJS even in an ESM package. A generic `entry` goes through the
  defaults and emits ESM.
- **Vite 8 bundles with Rolldown**: `build.rollupOptions` is silently ignored;
  it is `rolldownOptions`.
- **The grid's ResizeObserver needs the scroller to always exist.** An early
  `return` for the empty state meant the ref was null on the first render (the
  library is always empty then), so the observer was never attached and the
  effect — deps `[]` — never re-ran. The grid stayed at a zero-width viewport
  forever: one column, four cells, no matter the window size.
- **The token never crosses the context bridge.** The renderer asks main to make
  requests, so a compromised renderer cannot exfiltrate a credential it never
  held. `siftr://` exists for the same reason: an `<img>` cannot send an
  Authorization header, and a token in a query string would land in the DOM.
- **The server's read allowlist is rebuilt from the `roots` table at startup.**
  It used to live only in memory, so a library indexed from the CLI or in an
  earlier session made every thumbnail 403.
- **`webUtils.getPathForFile`** resolves dropped files; `File.path` was removed
  in Electron 32+ and it must be called from the preload.
- **Menu items dispatch to the renderer over `menu:action`** rather than acting
  in main. The renderer already owns library state; duplicating it would mean two
  definitions of what "Re-score" does. `menuTemplate()` is separated from
  `Menu.setApplicationMenu` so its shape is unit-testable — a menu that silently
  drops an item is otherwise only discoverable by opening it.
- **The icon is drawn, not sourced** (`build/make-icon.py`): filled polygons
  rather than thick strokes, because stroked diagonals leave mitre artifacts
  where they meet and the mark has to survive down to 16px. Re-run the script
  after editing it; it emits both the PNG and the `.icns`.
- **`vite.config.ts` derives `__dirname` from `import.meta.url`.** Vite's native
  config loader, due to become the default, does not provide `__dirname`.

## Testing

`pytest` — 339 tests, hermetic. It never downloads CLIP or InsightFace: a
deterministic colour-based `FakeEmbedder` and a `StubAnalyzer` stand in
(`tests/conftest.py`, `tests/test_faces.py`), because the logic worth testing
(storage, thresholds, reductions, placement, arg parsing) is independent of which
model produced the vectors. That keeps CI offline and sub-second.

`cd desktop && npx vitest run` — 17 tests over the pure ANY/ALL filter and
drop-payload logic.

Real-model verification is manual. What was checked on a synthetic 21-image
library (7 bicycles, 7 sunsets, 7 blueprints) plus two 3s videos:

- `teach` with negatives → threshold 0.504, cohesion 0.994
- `search -c bicycles` → exactly the 7 bike images + the bike video, top score
  0.995, no false positives; the sunset video correctly excluded
- `search -t "an architectural blueprint drawing"` → the 4 blueprints ranked first
- video sampling via the ffmpeg fallback → 4 frames per clip, one hit per file
- re-index → 21 unchanged, 0 re-embedded
- InsightFace `buffalo_l` loads and runs inference (0 faces on a blank image)

The desktop app was driven headlessly over the DevTools Protocol
(`SIFTR_DEBUG_PORT`, mirroring photo-slap's `PHOTO_SLAP_DEBUG_PORT`): tags and
counts render, 21/21 thumbnails load through `siftr://`, ANY gives the union
(14) and ALL the intersection (0), an override drops one file from an ALL result
(7 -> 6), and dropping 7 files on the new-tag target created the tag, scored, and
renamed all 7 — while removing a now-stale `[wheels]` bracket from another file.

## Packaging

`cd desktop && npm run dist` produces an unsigned DMG (117 MB) via
electron-builder. It bundles the UI but **not** Python. That is a deliberate
call, not an omission: torch is ~590 MB and onnxruntime ~80 MB, so vendoring the
Python side turns a 117 MB DMG into roughly 1.5 GB before any model downloads
(the HF and InsightFace caches are another ~1.2 GB at first run).

Service discovery order is override → bundled → PATH probe
(`electron/pythonService.ts`):

- `SIFTR_BIN` wins. This is the escape hatch for a virtualenv install, which is
  the common case and is never on the PATH a GUI app inherits.
- `Contents/Resources/python/bin/siftr` if a build vendored one. `extraResources`
  copies `desktop/resources/python` there; a plain `.venv` will NOT work when
  copied in, because its `pyvenv.cfg` points at the interpreter it was made from.
- Then a fixed list of likely bin directories, because a Finder-launched app gets
  a minimal PATH without Homebrew, pyenv, or any venv.

When none is found the app shows install instructions rather than failing
silently. That message is held in `fatal` state, separate from `error`: with no
service every request fails too, and those generic failures would otherwise bury
the one message that says what to do.

## Not done yet

1. **Scoring still scans every embedding**, though now once for all tags rather
   than once per tag (`service.score_library` stacks prototypes into one matmul).
   Beyond roughly a few hundred thousand vectors it wants an ANN index
   (hnswlib/faiss); the linear scan is fine below that and has no dependencies.
2. **Face recognition is verified on real photographs** (10 public-domain
   portraits, since deleted). Detection fired on 10/10 at det_score 0.76-0.89;
   registering one person from two references then re-matching found both of
   their photos and none of the other six people. Separation was wide — lowest
   true match 0.645, highest non-match 0.150 — so `DEFAULT_MATCH_THRESHOLD` at
   0.38 sits in the middle of a 0.495 gap rather than near an edge. Clustering
   recovered both same-person pairs with no chaining between identities.

   Note that RetinaFace does **not** fire on synthetic or drawn faces; it is
   trained on photographs. Test fixtures must use real images or stub the
   analyzer, which is why the suite does the latter.
3. **No Python in the packaged app** — see Packaging above for why.
4. **Clustering is single-link**, which can chain two people together through an
   ambiguous face. It did not chain on the real-photo check above, but that was
   eight faces of eight people; a large library with relatives in it is a much
   harder case. Fine for proposing groups a human confirms; a proper
   agglomerative pass with a merge criterion would be more robust.
