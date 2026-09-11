from pathlib import Path

from siftr.media import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    _sample_times,
    classify,
    discover,
    load_image,
)


def test_classify_known_types():
    assert classify(Path("a.JPG")) == "image"
    assert classify(Path("a.mp4")) == "video"
    assert classify(Path("a.txt")) is None
    assert classify(Path("a")) is None


def test_image_and_video_extensions_do_not_overlap():
    assert not IMAGE_EXTENSIONS & VIDEO_EXTENSIONS


def test_discover_finds_images_recursively(tmp_path, make_images):
    make_images(tmp_path / "one", (10, 10, 10), count=2)
    make_images(tmp_path / "two" / "deep", (20, 20, 20), count=3)
    (tmp_path / "readme.txt").write_text("x")
    assert len(list(discover(tmp_path))) == 5


def test_discover_skips_hidden_paths(tmp_path, make_images):
    make_images(tmp_path / "visible", (10, 10, 10), count=2)
    make_images(tmp_path / ".hidden", (10, 10, 10), count=3)
    found = list(discover(tmp_path))
    assert len(found) == 2
    assert all(".hidden" not in str(f.path) for f in found)


def test_discover_skips_photos_library_packages(tmp_path, make_images):
    """Walking into an Apple Photos library would index its internal derivatives."""
    make_images(tmp_path / "Photos Library.photoslibrary" / "resources", (5, 5, 5), count=4)
    make_images(tmp_path / "real", (10, 10, 10), count=2)
    assert len(list(discover(tmp_path))) == 2


def test_discover_accepts_a_single_file(tmp_path, make_images):
    make_images(tmp_path, (10, 10, 10), count=1)
    found = list(discover(tmp_path / "img00.png"))
    assert len(found) == 1 and found[0].kind == "image"


def test_discover_non_recursive_stays_shallow(tmp_path, make_images):
    make_images(tmp_path, (10, 10, 10), count=2)
    make_images(tmp_path / "deep", (10, 10, 10), count=3)
    assert len(list(discover(tmp_path, recursive=False))) == 2


def test_discover_records_size_and_mtime(tmp_path, make_images):
    make_images(tmp_path, (10, 10, 10), count=1)
    media = next(iter(discover(tmp_path)))
    assert media.size > 0 and media.mtime_ns > 0


def test_load_image_returns_rgb(tmp_path, make_images):
    make_images(tmp_path, (10, 200, 30), count=1)
    assert load_image(tmp_path / "img00.png").mode == "RGB"


def test_sample_times_are_inset_from_both_ends():
    """t=0 is very often a black frame; the end is often a fade or unreadable."""
    times = _sample_times(100.0, 5)
    assert len(times) == 5
    assert times[0] > 0.0
    assert times[-1] < 100.0
    assert times == sorted(times)


def test_sample_times_single_sample_is_the_midpoint():
    assert _sample_times(10.0, 1) == [5.0]


def test_sample_times_handles_zero_duration():
    assert _sample_times(0.0, 8) == [0.0]


def test_discover_returns_absolute_paths(tmp_path, make_images, monkeypatch):
    """An index is read from a different working directory than it was written
    from as a matter of course; relative paths make it unreadable."""
    make_images(tmp_path / "lib", (10, 10, 10), count=2)
    monkeypatch.chdir(tmp_path)

    found = list(discover(Path("lib")))

    assert len(found) == 2
    for media in found:
        assert media.path.is_absolute(), media.path


def test_discover_resolves_a_relative_single_file(tmp_path, make_images, monkeypatch):
    make_images(tmp_path / "lib", (10, 10, 10), count=1)
    monkeypatch.chdir(tmp_path)
    found = list(discover(Path("lib/img00.png")))
    assert found and found[0].path.is_absolute()
