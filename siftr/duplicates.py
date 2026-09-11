"""Finding duplicate and near-duplicate media.

Why not CLIP, given every file already has an embedding: CLIP is *trained* to be
invariant to exactly what separates a duplicate from a similar photo. Measured on
a fixture of one photo plus resize/recompress/brightness/crop variants against
genuinely different shots of the same scene, CLIP's cosine separated the two
classes by **0.003** — unusable as a threshold. A perceptual hash separated them
by a wide margin. They answer different questions: "is this the same subject" and
"is this the same photograph".

Two kinds of duplicate, found two ways:

* **Exact** — identical bytes. A BLAKE2b digest settles it with no false
  positives, which matters because a person may delete based on this.
* **Near** — the same photograph after a resize, a re-encode, an edit. A DCT
  perceptual hash, compared by Hamming distance.

Hash size is not a free parameter. At 64 bits the same fixture gave **no**
separation (duplicates 0-2 bits, different photos 2-6); at 256 bits duplicates
landed at 0.8-14.9% of bits and different photos at 18.0-23.5%. The extra
resolution is what makes the question answerable at all.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

#: DCT input size and the low-frequency square kept from it. 64/16 yields 255
#: bits (256 coefficients less the DC term).
DCT_SIZE = 64
KEEP = 16
HASH_BITS = KEEP * KEEP - 1
HASH_BYTES = (KEEP * KEEP) // 8

#: Fraction of differing bits below which two images are considered the same
#: photograph. Deliberately below the 18% where genuinely different shots began
#: in testing, and comfortably above the 2.4% that resize, re-encode, brightness
#: and blur produced. A hard crop measured 14.9%, so raising this toward 0.16
#: trades precision for catching those.
DEFAULT_DISTANCE = 0.12

_DCT_BASIS: np.ndarray | None = None


def _basis() -> np.ndarray:
    """The DCT-II basis, built once. Small enough to matmul directly."""
    global _DCT_BASIS
    if _DCT_BASIS is None:
        k = np.arange(DCT_SIZE)
        basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * DCT_SIZE))
        basis[:, 0] *= 1 / np.sqrt(2)
        _DCT_BASIS = basis
    return _DCT_BASIS


def perceptual_hash(image: Image.Image) -> bytes:
    """A 256-bit DCT hash, packed into 32 bytes.

    Greyscale and downscaled first, so the hash is blind to colour grading and
    resolution; thresholded at the median of the low-frequency coefficients, so
    it is blind to overall brightness. The DC term is dropped because it carries
    only mean luminance.
    """
    pixels = np.asarray(
        image.convert("L").resize((DCT_SIZE, DCT_SIZE), Image.LANCZOS), dtype=np.float64
    )
    basis = _basis()
    coefficients = basis.T @ pixels @ basis
    block = coefficients[:KEEP, :KEEP].flatten()
    bits = block > np.median(block[1:])
    bits[0] = False  # DC carries no shape information
    return np.packbits(bits).tobytes()


def content_hash(path: Path, chunk: int = 1 << 20) -> bytes:
    """A digest of the file's bytes, for exact duplicates.

    Streamed rather than read whole: a library contains videos, and reading a
    4 GB file into memory to hash it would be its own bug.
    """
    digest = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.digest()


def hamming(packed: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Bit differences between one packed hash and a stack of them."""
    return np.bitwise_count(packed[None, :] ^ other).sum(axis=1)


@dataclass
class DuplicateGroup:
    """One set of files that are the same photograph."""

    kind: str  # 'exact' | 'near'
    paths: list[Path] = field(default_factory=list)
    #: The copy worth keeping: highest resolution, then largest file.
    keeper: Path | None = None
    #: Worst distance within the group, as a fraction of bits. 0.0 when exact.
    distance: float = 0.0

    @property
    def size(self) -> int:
        return len(self.paths)

    def redundant(self) -> list[Path]:
        return [p for p in self.paths if p != self.keeper]


#: Names that advertise themselves as copies. Checked before modification time,
#: which sounds authoritative but is not: some copy tools preserve mtime and
#: others reset it, whereas a file called "IMG_1 copy 2.png" is telling you
#: exactly what it is.
_COPY_MARKERS = (
    re.compile(r"\bcopy\b", re.I),
    re.compile(r"\(\d+\)\s*$"),
    re.compile(r"[-_ ]\d+\s*$"),
    re.compile(r"\bduplicate\b", re.I),
)


def _looks_like_a_copy(name: str) -> bool:
    stem = Path(name).stem
    return any(pattern.search(stem) for pattern in _COPY_MARKERS)


def _derived_from_another(stem: str, others: list[str]) -> bool:
    """Whether this name looks derived from another in the same group.

    `sunset_bright.png` and `sunset_small.png` both extend `sunset.png`; an edit
    or an export keeps the original stem and adds to it, so a stem that strictly
    contains another group member's stem is very likely the derived copy. This
    catches edit suffixes without needing a list of what people call them, and it
    stays quiet when the names are unrelated — `IMG_0001` and `Corfu sunset` have
    no prefix relationship, so neither is demoted and the other rules decide.
    """
    return any(other != stem and stem.startswith(other) for other in others)


def _pick_keeper(rows: list[dict]) -> Path:
    """Choose the copy worth keeping.

    Resolution first, deliberately: a heavily compressed 12 MP frame is a better
    master than a lossless 800px export of it, and the larger *file* is often the
    worse one. Then bytes, as a proxy for encoding quality at equal resolution.

    At equal resolution the name decides, and it decides before file size: a
    brightened export of a PNG is often the *larger* file, so size would pick the
    edit over the original. So a name that advertises itself as a copy loses
    first, then one that looks derived from another name in the group, then the
    longer name, and only then size, age and path.
    """
    stems = [Path(r["path"]).stem for r in rows]
    best = min(
        rows,
        key=lambda r: (
            -(r.get("pixels") or 0),
            _looks_like_a_copy(r["path"]),
            _derived_from_another(Path(r["path"]).stem, stems),
            len(Path(r["path"]).name),
            -(r.get("size") or 0),
            r.get("mtime_ns") or 0,
            str(r["path"]),
        ),
    )
    return Path(best["path"])


def group_exact(rows: list[dict]) -> list[DuplicateGroup]:
    """Group files with identical content hashes."""
    buckets: dict[bytes, list[dict]] = {}
    for row in rows:
        digest = row.get("content_hash")
        if digest:
            buckets.setdefault(digest, []).append(row)

    groups = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        groups.append(
            DuplicateGroup(
                kind="exact",
                paths=[Path(m["path"]) for m in members],
                keeper=_pick_keeper(members),
                distance=0.0,
            )
        )
    return groups


def group_near(
    rows: list[dict], distance: float = DEFAULT_DISTANCE, chunk: int = 256
) -> list[DuplicateGroup]:
    """Group files whose perceptual hashes are within ``distance``.

    All-pairs Hamming, chunked so the temporary never exceeds a few hundred MB:
    at 50k files this is ~47s, which is acceptable for a scan run on demand. An
    ANN or LSH index would be needed an order of magnitude beyond that.
    """
    usable = [r for r in rows if r.get("phash")]
    if len(usable) < 2:
        return []

    packed = np.vstack([np.frombuffer(r["phash"], dtype=np.uint8) for r in usable])
    cutoff = round(distance * HASH_BITS)

    parent = list(range(len(usable)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    worst: dict[int, int] = {}
    for start in range(0, len(usable), chunk):
        block = packed[start : start + chunk]
        distances = np.bitwise_count(block[:, None, :] ^ packed[None, :, :]).sum(axis=2)
        for offset, row in enumerate(distances):
            i = start + offset
            # Only j > i: the matrix is symmetric and the diagonal is self.
            for j in np.nonzero(row <= cutoff)[0]:
                j = int(j)
                if j <= i:
                    continue
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
                root = find(i)
                worst[root] = max(worst.get(root, 0), int(row[j]))

    clusters: dict[int, list[dict]] = {}
    for index, row in enumerate(usable):
        clusters.setdefault(find(index), []).append(row)

    groups = []
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        groups.append(
            DuplicateGroup(
                kind="near",
                paths=[Path(m["path"]) for m in members],
                keeper=_pick_keeper(members),
                distance=worst.get(root, 0) / HASH_BITS,
            )
        )
    groups.sort(key=lambda g: (-g.size, str(g.keeper)))
    return groups


def find_duplicates(rows: list[dict], distance: float = DEFAULT_DISTANCE) -> list[DuplicateGroup]:
    """Exact groups first, then near groups that add something new.

    A set of byte-identical files is also perceptually identical, so reporting it
    twice would be noise; near groups that only restate an exact one are dropped.
    """
    exact = group_exact(rows)
    seen: set[Path] = {p for g in exact for p in g.paths}

    near = []
    for group in group_near(rows, distance):
        if set(group.paths) <= seen:
            continue
        near.append(group)

    return exact + near
