"""Reading and writing ``[tag]`` markers in filenames.

siftr writes the tags a file matched into its name, so the library stays
browsable and sorted outside the app. Doing that safely takes more care than it
first appears:

* **Only siftr's own tags are managed.** mediate's renamer already puts brackets
  in these filenames (``[2]``, ``[site 3]``), and other tools may too. A bracket
  group is rewritten only when its contents match a tag siftr knows about;
  everything else is left exactly where it is.
* **Applying tags is idempotent.** Re-scoring a library must converge, not append
  ``[glaze]`` a second time, and a tag that no longer matches must lose its
  bracket.
* **Order is canonical.** Managed tags are emitted sorted and last, so two runs
  that produce the same tag set produce byte-identical names regardless of what
  order the scorer happened to return them in.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

#: A bracket group and any run of spaces before it. Captured so a stem can be
#: split into unmanaged text and the groups siftr may rewrite.
_BRACKET = re.compile(r"\s*\[([^\[\]]*)\]")

#: Characters a tag may not contain: bracket syntax would nest, and the path
#: separator would change where the file lives.
_ILLEGAL = set("[]/\\")


class InvalidTagName(ValueError):
    """Raised for a tag that cannot be represented in a filename."""


def validate_tag(name: str) -> str:
    """Normalize a tag name, or explain why it cannot be one."""
    cleaned = " ".join(name.split())
    if not cleaned:
        raise InvalidTagName("tag name is empty")
    if any(ch in _ILLEGAL for ch in cleaned):
        raise InvalidTagName(f"tag name may not contain any of [ ] / \\ — got {name!r}")
    if cleaned.startswith("."):
        raise InvalidTagName(f"tag name may not start with a dot — got {name!r}")
    return cleaned


def parse_stem(stem: str, known: Iterable[str]) -> tuple[str, list[str]]:
    """Split a filename stem into its base text and siftr's tags.

    Bracket groups whose contents match a known tag (case-insensitively) are
    extracted; every other bracket group stays in the base text, in place.

    >>> parse_stem("beach day [2] [glaze]", ["glaze"])
    ('beach day [2]', ['glaze'])
    """
    lookup = {t.casefold(): t for t in known}
    found: list[str] = []

    def take(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        canonical = lookup.get(inner.casefold())
        if canonical is None:
            # Not ours — put the group back exactly as it was written.
            return match.group(0)
        if canonical not in found:
            found.append(canonical)
        return ""

    base = _BRACKET.sub(take, stem)
    return base.strip(), found


def format_stem(base: str, tags: Sequence[str]) -> str:
    """Reattach managed tags to a base stem, sorted and last.

    >>> format_stem("beach day [2]", ["outdoor", "glaze"])
    'beach day [2] [glaze] [outdoor]'
    """
    cleaned = base.strip()
    if not tags:
        return cleaned
    ordered = sorted(dict.fromkeys(tags), key=str.casefold)
    suffix = " ".join(f"[{t}]" for t in ordered)
    return f"{cleaned} {suffix}".strip()


def retag_name(filename: str, tags: Sequence[str], known: Iterable[str]) -> str:
    """The filename ``filename`` should have once it carries exactly ``tags``.

    ``known`` must include every tag siftr manages, not just the ones being
    applied — otherwise a tag that stopped matching would not be recognized as
    siftr's and its bracket would be left behind forever.
    """
    path = Path(filename)
    # Path.suffix on "archive.tar.gz" is ".gz", which is what we want: only the
    # final extension is preserved and the rest stays part of the stem.
    base, _existing = parse_stem(path.stem, known)
    new_stem = format_stem(base, tags)
    if not new_stem:
        # Refuse to produce a name that is nothing but an extension.
        new_stem = path.stem
    return new_stem + path.suffix


def tags_in_name(filename: str, known: Iterable[str]) -> list[str]:
    """The siftr tags currently written into a filename."""
    return parse_stem(Path(filename).stem, known)[1]


def needs_rename(filename: str, tags: Sequence[str], known: Iterable[str]) -> bool:
    """Whether applying ``tags`` would actually change the name."""
    return retag_name(filename, tags, known) != filename
