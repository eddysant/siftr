from siftr.organize import place


def test_symlink_leaves_the_original_in_place(tmp_path, make_images):
    make_images(tmp_path / "src", (10, 10, 10), count=3)
    sources = sorted((tmp_path / "src").glob("*.png"))

    result = place(sources, tmp_path / "out")
    assert result.placed == 3
    assert all(s.exists() for s in sources)
    assert all((tmp_path / "out" / s.name).is_symlink() for s in sources)


def test_copy_duplicates_the_file(tmp_path, make_images):
    make_images(tmp_path / "src", (10, 10, 10), count=1)
    source = next((tmp_path / "src").glob("*.png"))
    place([source], tmp_path / "out", mode="copy")
    target = tmp_path / "out" / source.name
    assert source.exists() and target.is_file() and not target.is_symlink()


def test_move_relocates_the_file(tmp_path, make_images):
    make_images(tmp_path / "src", (10, 10, 10), count=1)
    source = next((tmp_path / "src").glob("*.png"))
    place([source], tmp_path / "out", mode="move")
    assert not source.exists()
    assert (tmp_path / "out" / source.name).is_file()


def test_same_basename_from_two_folders_does_not_overwrite(tmp_path, make_images):
    """Libraries are full of distinct files named IMG_0001.jpg."""
    make_images(tmp_path / "a", (200, 10, 10), count=1)
    make_images(tmp_path / "b", (10, 10, 200), count=1)

    result = place(
        [tmp_path / "a" / "img00.png", tmp_path / "b" / "img00.png"],
        tmp_path / "out",
        mode="copy",
    )
    assert result.placed == 2
    assert (tmp_path / "out" / "img00.png").exists()
    assert (tmp_path / "out" / "img00-2.png").exists()


def test_dry_run_writes_nothing(tmp_path, make_images):
    make_images(tmp_path / "src", (10, 10, 10), count=2)
    sources = sorted((tmp_path / "src").glob("*.png"))
    result = place(sources, tmp_path / "out", dry_run=True)
    assert result.placed == 2
    assert not (tmp_path / "out").exists()


def test_missing_sources_are_skipped_not_fatal(tmp_path, make_images):
    make_images(tmp_path / "src", (10, 10, 10), count=1)
    real = next((tmp_path / "src").glob("*.png"))
    result = place([real, tmp_path / "src" / "ghost.png"], tmp_path / "out")
    assert result.placed == 1 and result.skipped == 1
