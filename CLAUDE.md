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

## Testing

`pytest` — 100 tests, hermetic. It never downloads CLIP or InsightFace: a
deterministic colour-based `FakeEmbedder` and a `StubAnalyzer` stand in
(`tests/conftest.py`, `tests/test_faces.py`), because the logic worth testing
(storage, thresholds, reductions, placement, arg parsing) is independent of which
model produced the vectors. That keeps CI offline and sub-second.

Real-model verification is manual. What was checked on a synthetic 21-image
library (7 bicycles, 7 sunsets, 7 blueprints) plus two 3s videos:

- `teach` with negatives → threshold 0.504, cohesion 0.994
- `search -c bicycles` → exactly the 7 bike images + the bike video, top score
  0.995, no false positives; the sunset video correctly excluded
- `search -t "an architectural blueprint drawing"` → the 4 blueprints ranked first
- video sampling via the ffmpeg fallback → 4 frames per clip, one hit per file
- re-index → 21 unchanged, 0 re-embedded
- InsightFace `buffalo_l` loads and runs inference (0 faces on a blank image)

## Not done yet

1. **No face *clustering*.** Unidentified faces are stored but never grouped, so
   there is no "who is this person who appears 40 times?" flow. The embeddings
   are already in `file_faces`; this is a query plus a clustering pass.
2. **`apply_concept` rescans every embedding.** Fine to ~100k vectors; beyond
   that it wants an ANN index (hnswlib/faiss) rather than a full scan.
3. **No incremental video re-sampling.** Changing `--video-samples` needs
   `--force` on the whole library.
4. **HEIC depends on Pillow's HEIF support**, which is not present in every
   wheel. Untested here.
