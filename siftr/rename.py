"""Applying ``[tag]`` markers to files on disk, reversibly.

Scoring the library renames matching files automatically. Two things make that
safe rather than reckless:

* **Every batch is journaled** to ``.siftr-renames.json`` at the library root and
  can be replayed backwards, the same pattern mediate uses for its own renamer.
* **Nothing is ever overwritten.** A target that already exists gets a numeric
  suffix, and a rename whose target is another pending rename's source waits a
  round rather than clobbering it.

The index stores absolute paths, so a rename that did not also update the
database would orphan every embedding for that file. Both happen together here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .db import Database
from .naming import retag_name

MANIFEST_NAME = ".siftr-renames.json"


@dataclass
class RenamePlan:
    source: Path
    target: Path

    @property
    def is_case_only(self) -> bool:
        """A rename differing only in case, which needs care on APFS/NTFS."""
        return (
            self.source != self.target
            and str(self.source).casefold() == str(self.target).casefold()
        )


@dataclass
class RenameResult:
    applied: list[tuple[Path, Path]] = field(default_factory=list)
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.applied)


def plan_renames(
    tagged: Iterable[tuple[Path, Sequence[str]]], known: Iterable[str]
) -> list[RenamePlan]:
    """Work out which files need renaming for the tags they now carry."""
    known = list(known)
    plans: list[RenamePlan] = []
    for path, tags in tagged:
        path = Path(path)
        new_name = retag_name(path.name, list(tags), known)
        if new_name != path.name:
            plans.append(RenamePlan(path, path.with_name(new_name)))
    return plans


def _exists(path: Path) -> bool:
    """Whether a path is taken.

    ``Path.exists()`` follows symlinks and returns False for a broken one, which
    would let a rename clobber it; ``lstat`` is what actually answers "is this
    name in use".
    """
    try:
        os.lstat(path)
    except (OSError, ValueError):
        return False
    return True


def _free_target(target: Path, claimed: set[Path]) -> Path:
    """A target name that is free on disk and not already claimed this batch."""
    if not _exists(target) and target not in claimed:
        return target
    stem, suffix = target.stem, target.suffix
    counter = 2
    while True:
        candidate = target.with_name(f"{stem}-{counter}{suffix}")
        if not _exists(candidate) and candidate not in claimed:
            return candidate
        counter += 1


def apply_renames(
    plans: Sequence[RenamePlan],
    db: Database | None = None,
    root: Path | None = None,
    dry_run: bool = False,
) -> RenameResult:
    """Execute rename plans, updating the index and journaling the batch.

    Plans whose target is another plan's source are deferred: renaming
    ``a -> b`` while ``b -> c`` is still pending would destroy ``b``. Each round
    applies whatever is currently unblocked; if a full round makes no progress
    the remainder is skipped rather than forced.
    """
    result = RenameResult()
    pending = [p for p in plans if p.source != p.target]
    if not pending:
        return result

    claimed: set[Path] = set()

    while pending:
        sources = {p.source for p in pending}
        # A plan is blocked while its target is still occupied by a file that is
        # itself waiting to move out of the way.
        ready = [p for p in pending if p.target not in sources or p.is_case_only]
        if not ready:
            result.skipped += len(pending)
            result.errors.append(f"{len(pending)} rename(s) skipped: circular target dependency")
            break

        for plan in ready:
            target = plan.target
            if not plan.is_case_only:
                target = _free_target(target, claimed)

            if dry_run:
                claimed.add(target)
                result.applied.append((plan.source, target))
                continue

            try:
                if not _exists(plan.source):
                    result.skipped += 1
                    continue
                # os.replace would overwrite; os.rename on a case-insensitive
                # filesystem is the correct call for a case-only change, where
                # source and target are "the same file".
                os.rename(plan.source, target)
            except OSError as exc:
                result.errors.append(f"{plan.source}: {exc}")
                result.skipped += 1
                continue

            claimed.add(target)
            result.applied.append((plan.source, target))
            if db is not None:
                db.rename_file(plan.source, target)

        pending = [p for p in pending if p not in ready]

    if db is not None and not dry_run and result.applied:
        db.commit()
    if root is not None and not dry_run and result.applied:
        _journal(root, result.applied)
    return result


def _journal(root: Path, applied: Sequence[tuple[Path, Path]]) -> None:
    """Append one batch to the undo manifest.

    Written via a temp file and ``os.replace`` so an interrupted write cannot
    truncate the history of every previous batch.
    """
    manifest = Path(root) / MANIFEST_NAME
    history = read_manifest(root)
    history.append({"renames": [[str(s), str(t)] for s, t in applied]})

    tmp = manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
    os.replace(tmp, manifest)


def read_manifest(root: Path) -> list[dict]:
    manifest = Path(root) / MANIFEST_NAME
    if not manifest.exists():
        return []
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def undo_last(root: Path, db: Database | None = None, dry_run: bool = False) -> RenameResult:
    """Reverse the most recent rename batch."""
    history = read_manifest(root)
    if not history:
        return RenameResult()

    batch = history[-1]
    reversed_plans = [
        RenamePlan(Path(target), Path(source))
        # Reverse within the batch too: the forward pass may have resolved a
        # collision by suffixing, and undoing in reverse order frees names in
        # the opposite order they were taken.
        for source, target in reversed(batch.get("renames", []))
    ]
    result = apply_renames(reversed_plans, db=db, root=None, dry_run=dry_run)

    if not dry_run and result.applied:
        history.pop()
        manifest = Path(root) / MANIFEST_NAME
        if history:
            tmp = manifest.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
            os.replace(tmp, manifest)
        else:
            manifest.unlink(missing_ok=True)
    return result
