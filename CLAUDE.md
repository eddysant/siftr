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
| `index.py` | the indexing pass, `apply_concept`, `rematch_faces` |
| `search.py` | ranking by concept / examples / text / person |
| `organize.py` | symlink/copy/move results into folders |
| `vectors.py` | normalize, blob (de)serialize, cosine, centroid |

## Design decisions worth keeping

- **Every stored embedding is L2-normalized** (`vectors.py`), so a dot product
  *is* cosine similarity. The invariant lives in one module; nothing else
  handles magnitudes.
- **Prototype, not a trained head.** A concept is the normalized mean of its
  examples. With 5–50 examples and no negatives there is not enough data to fit
  anything larger, and a prototype cannot overfit or diverge.
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

`pytest` — 267 tests, hermetic. It never downloads CLIP or InsightFace: a
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
2. **Face detection on real photographs is unverified here.** The models load and
   run, and every code path is covered with stub embeddings, but synthetic
   drawings are not detected by RetinaFace (it is trained on photos) and no real
   face images were used. Verify with `siftr add-person "Name" ~/some/photos`.
3. **No Python in the packaged app** — see Packaging above for why.
4. **Clustering is single-link**, which can chain two people together through an
   ambiguous face. Fine for proposing groups a human confirms; a proper
   agglomerative pass with a merge criterion would be more robust.
