"""Grouping files that must travel together.

A photo on disk is often not a single file:

* **Live Photos** are a still (``IMG_4821.HEIC``) plus a motion clip
  (``IMG_4821.MOV``) paired *by stem*. Rename or move one without the other and
  the pairing is silently destroyed.
* **Sidecars** — Apple's ``.AAE`` edit lists, Adobe's ``.XMP``, camera ``.THM``
  thumbnails — carry edits and metadata for a media file and are matched the same
  way. Leave one behind and the edits are orphaned.

siftr scores every media file independently, so the two halves of a Live Photo
routinely match different tags and would be given different names. Everything in
a group therefore takes its tags from one **primary** member, which keeps the
stems identical on both sides of any rename or move.

Grouping is by stem alone. ExifTool's ``ContentIdentifier`` would confirm a true
Live Photo pair, but grouping two unrelated same-stem files is harmless — they
simply keep matching stems — whereas failing to group a real pair breaks it. The
conservative default is therefore to group.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Non-media files that belong to the media file sharing their stem.
SIDECAR_EXTENSIONS = {".aae", ".xmp", ".thm", ".json", ".txt"}

#: Preferred primary, best first. The still image owns a Live Photo's identity,
#: so it decides the tags and the stem for the whole group.
_PRIMARY_ORDER = [
    ".heic",
    ".heif",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".mov",
    ".mp4",
    ".m4v",
    ".avi",
    ".mkv",
]


@dataclass
class Group:
    """One media file plus everything that must move with it."""

    primary: Path
    members: list[Path]

    @property
    def stem(self) -> str:
        return self.primary.stem

    @property
    def is_paired(self) -> bool:
        return len(self.members) > 1

    def sidecars(self) -> list[Path]:
        return [p for p in self.members if p != self.primary]


def _rank(path: Path) -> int:
    suffix = path.suffix.lower()
    return _PRIMARY_ORDER.index(suffix) if suffix in _PRIMARY_ORDER else len(_PRIMARY_ORDER)


def is_sidecar(path: Path) -> bool:
    return path.suffix.lower() in SIDECAR_EXTENSIONS


def group_paths(paths: Iterable[Path]) -> list[Group]:
    """Group paths by (directory, stem).

    Stems are compared case-insensitively: macOS and Windows filesystems treat
    ``IMG_1.HEIC`` and ``img_1.mov`` as the same stem, and a pair spelled that way
    is still a pair.
    """
    buckets: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in paths:
        path = Path(path)
        buckets[(str(path.parent), path.stem.casefold())].append(path)

    groups: list[Group] = []
    for members in buckets.values():
        ordered = sorted(members, key=lambda p: (_rank(p), str(p)))
        # A bucket of nothing but sidecars has no media file to own it; leave
        # those alone rather than inventing a primary and renaming them.
        media = [p for p in ordered if not is_sidecar(p)]
        if not media:
            continue
        groups.append(Group(primary=media[0], members=ordered))
    return groups


def find_companions(path: Path) -> list[Path]:
    """Every file on disk sharing ``path``'s stem, including ``path`` itself.

    Used when acting on files chosen individually rather than from a full scan,
    so a companion that was never indexed still travels with its primary.
    """
    path = Path(path)
    parent = path.parent
    if not parent.is_dir():
        return [path]

    stem = path.stem.casefold()
    found = [p for p in parent.iterdir() if p.is_file() and p.stem.casefold() == stem]
    return sorted(found, key=lambda p: (_rank(p), str(p))) or [path]


def tags_for_groups(
    groups: Sequence[Group], tags_by_path: dict[Path, list[str]]
) -> dict[Path, list[str]]:
    """Give every member of a group the primary's tags.

    This is what stops a Live Photo's still and motion clip being renamed apart.
    Using the primary's tags rather than the union is deliberate: the union would
    put tags on the still that only the motion clip matched, which is both
    surprising and unstable — a re-score of one half would change the other's
    name.
    """
    resolved: dict[Path, list[str]] = {}
    for group in groups:
        shared = tags_by_path.get(group.primary, [])
        for member in group.members:
            resolved[member] = list(shared)
    return resolved
