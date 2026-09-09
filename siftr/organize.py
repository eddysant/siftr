"""Materializing search results as folders on disk.

Symlinking is the default and copying is opt-in; moving is available but
deliberately awkward to reach. A media library is the user's originals, and a
tagger that reorganizes them by default is a tagger that eventually loses
something. Symlinks give you browsable folders while the originals stay put.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Mode = Literal["symlink", "copy", "move"]


@dataclass
class OrganizeResult:
    placed: int = 0
    skipped: int = 0
    errors: list[str] | None = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def place(
    paths: Iterable[Path],
    destination: Path,
    mode: Mode = "symlink",
    dry_run: bool = False,
) -> OrganizeResult:
    """Put ``paths`` into ``destination`` by symlink, copy, or move.

    Name collisions are resolved by appending a counter rather than overwriting,
    since two folders in a library commonly hold different files with the same
    basename (``IMG_0001.jpg``).
    """
    destination = Path(destination).expanduser()
    result = OrganizeResult()

    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)

    for source in paths:
        source = Path(source)
        if not source.exists():
            result.skipped += 1
            continue

        target = _free_name(destination, source.name)

        if dry_run:
            result.placed += 1
            continue

        try:
            if mode == "symlink":
                target.symlink_to(source.resolve())
            elif mode == "copy":
                shutil.copy2(source, target)
            elif mode == "move":
                shutil.move(str(source), str(target))
            else:
                raise ValueError(f"unknown mode: {mode}")
            result.placed += 1
        except OSError as exc:
            result.errors.append(f"{source}: {exc}")
            result.skipped += 1

    return result


def _free_name(folder: Path, name: str) -> Path:
    candidate = folder / name
    if not candidate.exists() and not candidate.is_symlink():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    counter = 2
    while True:
        candidate = folder / f"{stem}-{counter}{suffix}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        counter += 1
