"""Duplicate and near-duplicate detection.

The design claim under test: a perceptual hash separates "the same photograph"
from "a similar photograph", and CLIP does not — CLIP is trained to be invariant
to precisely the difference. Several of these assert that separation directly.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter

from siftr.duplicates import (
    DEFAULT_DISTANCE,
    HASH_BITS,
    HASH_BYTES,
    content_hash,
    find_duplicates,
    group_exact,
    group_near,
    perceptual_hash,
)


def photo(seed, jitter=0, size=(400, 300)):
    """A textured scene. Texture matters: a flat gradient is a pathological case
    for a DCT hash and would make any threshold look broken."""
    rng = np.random.default_rng(seed)
    width, height = size
    base = np.zeros((height, width, 3), dtype=np.int16)
    base[:, :, 0] = np.linspace(240, 150, height)[:, None]
    base[:, :, 1] = np.linspace(200, 160, height)[:, None]
    base[:, :, 2] = np.linspace(150, 210, height)[:, None]
    for i in range(6):
        x = 30 + i * 60 + (rng.integers(-25, 25) if jitter else 0)
        h = 60 + (rng.integers(-20, 20) if jitter else 0)
        base[max(0, height - 100 - h) : height - 100, x : x + 18] = (40 + i * 8, 90, 60)
    base += rng.integers(-22, 23, base.shape, dtype=np.int16)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def row(path, image=None, **extra):
    image = image or Image.open(path).convert("RGB")
    return {
        "path": str(path),
        "content_hash": content_hash(path),
        "phash": perceptual_hash(image),
        "pixels": image.size[0] * image.size[1],
        "size": path.stat().st_size,
        **extra,
    }


# ------------------------------------------------------------------- hashing


def test_hash_is_the_expected_width():
    assert len(perceptual_hash(photo(1))) == HASH_BYTES
    assert HASH_BITS == 255


def test_hash_is_deterministic():
    image = photo(1)
    assert perceptual_hash(image) == perceptual_hash(image)


def test_hash_survives_resize_recompress_and_brightness(tmp_path):
    original = photo(1, size=(800, 600))
    base = perceptual_hash(original)

    for label, variant in [
        ("resized", original.resize((400, 300))),
        ("brighter", ImageEnhance.Brightness(original).enhance(1.15)),
        ("blurred", original.filter(ImageFilter.GaussianBlur(1.0))),
    ]:
        differing = np.unpackbits(
            np.frombuffer(base, dtype=np.uint8)
            ^ np.frombuffer(perceptual_hash(variant), dtype=np.uint8)
        ).sum()
        assert differing / HASH_BITS < DEFAULT_DISTANCE, f"{label} should still match"


def test_hash_distinguishes_a_genuinely_different_photo():
    base = perceptual_hash(photo(1))
    other = perceptual_hash(photo(1, jitter=1))
    differing = np.unpackbits(
        np.frombuffer(base, dtype=np.uint8) ^ np.frombuffer(other, dtype=np.uint8)
    ).sum()
    assert differing / HASH_BITS > DEFAULT_DISTANCE, "moved subjects are not a duplicate"


def test_content_hash_matches_only_identical_bytes(tmp_path):
    a = tmp_path / "a.png"
    photo(1).save(a)
    b = tmp_path / "b.png"
    shutil.copy(a, b)
    c = tmp_path / "c.png"
    photo(2).save(c)

    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash(c)


def test_content_hash_streams_large_files(tmp_path):
    """Reading a 4 GB video whole to hash it would be its own bug."""
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (5 << 20))
    assert len(content_hash(big, chunk=4096)) == 16


# -------------------------------------------------------------------- exact


def test_exact_groups_identical_files(tmp_path):
    a = tmp_path / "a.png"
    photo(1).save(a)
    b = tmp_path / "copy.png"
    shutil.copy(a, b)
    c = tmp_path / "other.png"
    photo(2).save(c)

    groups = group_exact([row(a), row(b), row(c)])
    assert len(groups) == 1
    assert {p.name for p in groups[0].paths} == {"a.png", "copy.png"}
    assert groups[0].distance == 0.0


def test_a_lone_file_is_not_a_group(tmp_path):
    a = tmp_path / "a.png"
    photo(1).save(a)
    assert group_exact([row(a)]) == []


def test_files_without_a_hash_are_ignored(tmp_path):
    a = tmp_path / "a.png"
    photo(1).save(a)
    rows = [row(a), {"path": "ghost.png", "content_hash": None, "phash": None}]
    assert group_exact(rows) == []


# --------------------------------------------------------------------- near


def test_near_groups_variants_of_one_photo(tmp_path):
    original = photo(1, size=(800, 600))
    paths = []
    for name, image in [
        ("original.png", original),
        ("resized.png", original.resize((400, 300))),
        ("brighter.png", ImageEnhance.Brightness(original).enhance(1.15)),
    ]:
        p = tmp_path / name
        image.save(p)
        paths.append(p)
    different = tmp_path / "different.png"
    photo(1, jitter=1, size=(800, 600)).save(different)

    groups = group_near([row(p) for p in [*paths, different]])

    assert len(groups) == 1
    assert {p.name for p in groups[0].paths} == {"original.png", "resized.png", "brighter.png"}


def test_near_keeps_the_highest_resolution_copy(tmp_path):
    original = photo(1, size=(800, 600))
    big = tmp_path / "big.png"
    original.save(big)
    small = tmp_path / "small.png"
    original.resize((200, 150)).save(small)

    groups = group_near([row(big), row(small)])
    assert groups[0].keeper.name == "big.png"
    assert [p.name for p in groups[0].redundant()] == ["small.png"]


def test_unrelated_photos_are_not_grouped(tmp_path):
    paths = []
    for i in range(4):
        p = tmp_path / f"{i}.png"
        photo(100 + i, jitter=1).save(p)
        paths.append(p)
    assert group_near([row(p) for p in paths]) == []


def test_a_tighter_distance_finds_fewer_groups(tmp_path):
    original = photo(1, size=(800, 600))
    a = tmp_path / "a.png"
    original.save(a)
    b = tmp_path / "b.png"
    original.filter(ImageFilter.GaussianBlur(1.0)).save(b)

    assert len(group_near([row(a), row(b)], distance=DEFAULT_DISTANCE)) == 1
    assert group_near([row(a), row(b)], distance=0.0) == []


def test_grouping_is_transitive(tmp_path):
    """A chain of near-identical files belongs in one group, not several."""
    original = photo(1, size=(800, 600))
    rows = []
    for i, scale in enumerate([1.0, 0.9, 0.8, 0.7]):
        p = tmp_path / f"{i}.png"
        original.resize((int(800 * scale), int(600 * scale))).save(p)
        rows.append(row(p))

    groups = group_near(rows)
    assert len(groups) == 1
    assert groups[0].size == 4


@pytest.mark.parametrize("chunk", [1, 2, 1000])
def test_chunking_does_not_change_the_result(tmp_path, chunk):
    original = photo(1, size=(800, 600))
    rows = []
    for i, scale in enumerate([1.0, 0.9, 0.8]):
        p = tmp_path / f"{i}.png"
        original.resize((int(800 * scale), int(600 * scale))).save(p)
        rows.append(row(p))
    groups = group_near(rows, chunk=chunk)
    assert len(groups) == 1 and groups[0].size == 3


# ------------------------------------------------------------------ combined


def test_exact_and_near_are_both_reported(tmp_path):
    original = photo(1, size=(800, 600))
    a = tmp_path / "a.png"
    original.save(a)
    exact = tmp_path / "exact.png"
    shutil.copy(a, exact)
    resized = tmp_path / "resized.png"
    original.resize((400, 300)).save(resized)

    groups = find_duplicates([row(a), row(exact), row(resized)])
    kinds = {g.kind for g in groups}
    assert "exact" in kinds and "near" in kinds


def test_a_near_group_that_only_restates_an_exact_one_is_dropped(tmp_path):
    """Byte-identical files are also perceptually identical; saying so twice is
    noise."""
    a = tmp_path / "a.png"
    photo(1).save(a)
    b = tmp_path / "b.png"
    shutil.copy(a, b)

    groups = find_duplicates([row(a), row(b)])
    assert [g.kind for g in groups] == ["exact"]


def test_no_duplicates_is_an_empty_result(tmp_path):
    rows = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        photo(200 + i, jitter=1).save(p)
        rows.append(row(p))
    assert find_duplicates(rows) == []


def test_empty_input(tmp_path):
    assert find_duplicates([]) == []


# --------------------------------------------------- the threshold itself


def _distance(a, b):
    return (
        np.unpackbits(
            np.frombuffer(perceptual_hash(a), dtype=np.uint8)
            ^ np.frombuffer(perceptual_hash(b), dtype=np.uint8)
        ).sum()
        / HASH_BITS
    )


def test_default_distance_sits_between_the_two_populations():
    """Pins the value, not just the behaviour.

    Measured on these fixtures: re-encodes, resizes, brightness and blur land at
    1.6-2.4% of bits, while different photographs of the same scene start around
    18%. A threshold outside that band is wrong in a way the behavioural tests
    above will not always catch — too loose and genuinely different photos get
    proposed for deletion, which is the expensive direction to be wrong in.
    """
    assert 0.05 <= DEFAULT_DISTANCE <= 0.16


def test_a_photo_differing_only_slightly_is_still_not_a_duplicate():
    """The hard negative: same scene, subjects nudged rather than rearranged."""
    base = photo(1, size=(800, 600))
    nudged = photo(1, jitter=1, size=(800, 600))

    separation = _distance(base, nudged)
    assert separation > DEFAULT_DISTANCE, (
        f"a different photo at {separation:.1%} must not fall inside the threshold"
    )
    # And the margin must be real, not a rounding accident.
    assert separation - DEFAULT_DISTANCE > 0.05


def test_duplicates_sit_far_below_the_threshold():
    base = photo(1, size=(800, 600))
    for variant in (
        base.resize((400, 300)),
        ImageEnhance.Brightness(base).enhance(1.15),
        base.filter(ImageFilter.GaussianBlur(1.0)),
    ):
        assert _distance(base, variant) < DEFAULT_DISTANCE / 2


def test_keeper_prefers_the_original_over_a_copy(tmp_path):
    """Same resolution, same bytes: the tiebreaks have to carry this."""
    image = photo(1, size=(400, 300))
    original = tmp_path / "original.png"
    image.save(original)
    copy = tmp_path / "original_copy.png"
    shutil.copy(original, copy)

    rows = [
        {**row(original), "mtime_ns": 1_000},
        {**row(copy), "mtime_ns": 2_000},
    ]
    assert group_exact(rows)[0].keeper.name == "original.png"


def test_keeper_prefers_resolution_over_file_size(tmp_path):
    """A compressed 12 MP frame beats a lossless thumbnail of it."""
    image = photo(1, size=(800, 600))
    big = tmp_path / "big.jpg"
    image.save(big, quality=30)
    small = tmp_path / "small.png"
    image.resize((200, 150)).save(small)

    assert big.stat().st_size < small.stat().st_size, "fixture must have the big file smaller"
    assert group_near([row(big), row(small)])[0].keeper.name == "big.jpg"


def test_keeper_falls_back_to_the_shorter_name(tmp_path):
    image = photo(1, size=(400, 300))
    plain = tmp_path / "IMG_1.png"
    image.save(plain)
    suffixed = tmp_path / "IMG_1 copy 2.png"
    shutil.copy(plain, suffixed)

    rows = [{**row(plain), "mtime_ns": 5}, {**row(suffixed), "mtime_ns": 5}]
    assert group_exact(rows)[0].keeper.name == "IMG_1.png"


@pytest.mark.parametrize(
    "copy_name",
    ["IMG_1 copy.png", "IMG_1 copy 2.png", "IMG_1 (1).png", "IMG_1-2.png", "IMG_1 duplicate.png"],
)
def test_keeper_demotes_names_that_advertise_being_a_copy(tmp_path, copy_name):
    """mtime is unreliable — copy tools differ — but the name is explicit."""
    image = photo(1, size=(400, 300))
    original = tmp_path / "IMG_1.png"
    image.save(original)
    duplicate = tmp_path / copy_name
    shutil.copy(original, duplicate)

    # The copy given the *older* mtime, so only the name can save the original.
    rows = [
        {**row(original), "mtime_ns": 9_000},
        {**row(duplicate), "mtime_ns": 1_000},
    ]
    assert group_exact(rows)[0].keeper.name == "IMG_1.png"


def test_keeper_prefers_the_original_over_a_derived_edit(tmp_path):
    """A brightened export is often the *larger* PNG, so size alone picks the
    edit. The name says which came first: an edit extends the original's stem."""
    original = tmp_path / "sunset.png"
    photo(1, size=(800, 600)).save(original)
    edit = tmp_path / "sunset_bright.png"
    ImageEnhance.Brightness(photo(1, size=(800, 600))).enhance(1.12).save(edit)

    rows = [row(original), row(edit)]
    assert group_near(rows, distance=0.25)[0].keeper.name == "sunset.png"


def test_unrelated_names_are_not_demoted(tmp_path):
    """The prefix rule must stay quiet when the names have nothing to do with
    each other, or it would pick arbitrarily."""
    from siftr.duplicates import _derived_from_another

    stems = ["IMG_0001", "Corfu sunset"]
    assert not _derived_from_another("IMG_0001", stems)
    assert not _derived_from_another("Corfu sunset", stems)


def test_a_derived_name_is_recognised():
    from siftr.duplicates import _derived_from_another

    stems = ["sunset", "sunset_bright", "sunset_small"]
    assert not _derived_from_another("sunset", stems)
    assert _derived_from_another("sunset_bright", stems)
    assert _derived_from_another("sunset_small", stems)


def test_resolution_still_beats_the_name_rule(tmp_path):
    """A high-res edit is still a better master than a low-res original.

    Exercises the keeper rule directly: two renditions at different sizes do not
    reliably group as near-duplicates anyway, and grouping is not what is under
    test here.
    """
    from siftr.duplicates import _pick_keeper

    small = tmp_path / "sunset.png"
    photo(1, size=(200, 150)).save(small)
    big = tmp_path / "sunset_bright.png"
    photo(1, size=(800, 600)).save(big)

    assert _pick_keeper([row(small), row(big)]).name == "sunset_bright.png"


def test_derived_name_rule_outranks_plain_name_length(tmp_path):
    """The case where the two rules disagree.

    A shorter *filename* is only a proxy for "not derived from something else".
    Here the derived copy has the shorter name — a short extension against a long
    one — so name length alone would keep the edit. The prefix relationship
    between the stems is the thing that actually says which came first.
    """
    from siftr.duplicates import _pick_keeper

    original = tmp_path / "sunset.jpeg"  # 11 characters
    photo(1, size=(400, 300)).save(original)
    derived = tmp_path / "sunsetX.png"  # 11 characters, but stem extends "sunset"
    photo(1, size=(400, 300)).save(derived)

    rows = [
        {**row(original), "size": 10, "mtime_ns": 1},
        {**row(derived), "size": 999, "mtime_ns": 1},
    ]
    assert _pick_keeper(rows).name == "sunset.jpeg"


# ------------------------------------------------------------- collecting


def _library_with_dupes(db, tmp_path):
    """Two identical files plus an unrelated one, indexed."""

    lib = tmp_path / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    photo(1, size=(200, 150)).save(lib / "original.png")
    shutil.copy(lib / "original.png", lib / "original copy.png")
    photo(99, jitter=1, size=(200, 150)).save(lib / "other.png")
    return lib


def test_collecting_leaves_the_keeper_alone(db, tmp_path, embedder):
    """This gathers what you *could* remove; the library stays intact."""
    from siftr.index import build_index
    from siftr.service import collect_duplicates

    lib = _library_with_dupes(db, tmp_path)
    build_index(db, lib, embedder, detect_faces=False)

    result = collect_duplicates(db, tmp_path / "review")

    assert result["collected"] == 1
    assert (lib / "original.png").exists(), "the keeper is never collected"
    assert (lib / "original copy.png").exists(), "symlinking leaves the original"
    assert (tmp_path / "review" / "original copy.png").is_symlink()


def test_collecting_nothing_is_not_an_error(db, tmp_path, make_images, embedder):
    from siftr.index import build_index
    from siftr.service import collect_duplicates

    make_images(tmp_path / "lib", (200, 40, 40), count=1)
    build_index(db, tmp_path / "lib", embedder, detect_faces=False)
    assert collect_duplicates(db, tmp_path / "review")["collected"] == 0


def test_collecting_by_move_keeps_the_index_pointing_at_the_files(db, tmp_path, embedder):
    """Moving changes where files live; a stale index points nowhere."""
    from siftr.index import build_index
    from siftr.service import collect_duplicates

    lib = _library_with_dupes(db, tmp_path)
    build_index(db, lib, embedder, detect_faces=False)

    collect_duplicates(db, tmp_path / "review", mode="move")

    paths = [r["path"] for r in db.conn.execute("SELECT path FROM files")]
    for path in paths:
        assert Path(path).exists(), f"index points at a file that is not there: {path}"


def test_collecting_dry_run_writes_nothing(db, tmp_path, embedder):
    from siftr.index import build_index
    from siftr.service import collect_duplicates

    lib = _library_with_dupes(db, tmp_path)
    build_index(db, lib, embedder, detect_faces=False)

    result = collect_duplicates(db, tmp_path / "review", dry_run=True)

    assert result["collected"] == 1
    assert not (tmp_path / "review").exists()


def test_a_looser_distance_collects_more(db, tmp_path, embedder):
    from siftr.index import build_index
    from siftr.service import collect_duplicates

    lib = tmp_path / "lib"
    lib.mkdir(parents=True)
    base = photo(1, size=(800, 600))
    base.save(lib / "a.png")
    base.resize((400, 300)).save(lib / "a_small.png")
    build_index(db, lib, embedder, detect_faces=False)

    tight = collect_duplicates(db, tmp_path / "r1", distance=0.0, dry_run=True)
    loose = collect_duplicates(db, tmp_path / "r2", distance=0.30, dry_run=True)
    assert loose["collected"] > tight["collected"]
