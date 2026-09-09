"""CLI argument parsing and the commands that need no model."""

from pathlib import Path

import pytest

from siftr.cli import build_parser, main, resolve_globals


def parse(argv):
    return build_parser().parse_args(argv)


def globals_for(argv):
    return resolve_globals(parse(argv))


def test_global_flags_before_the_subcommand():
    db, quiet = globals_for(["-q", "--db", "/tmp/x.db", "status"])
    assert quiet is True
    assert db == Path("/tmp/x.db")


def test_global_flags_after_the_subcommand():
    """`siftr index ~/Pictures -q` is the natural way to type it."""
    db, quiet = globals_for(["index", "lib", "-q", "--db", "/tmp/x.db"])
    assert quiet is True
    assert db == Path("/tmp/x.db")


def test_global_flags_default_when_absent():
    db, quiet = globals_for(["status"])
    assert quiet is False
    assert db.name == "index.db"


def test_subcommand_does_not_clobber_a_flag_given_earlier():
    """argparse's subparser copies a fresh namespace over the parent's."""
    assert globals_for(["-q", "index", "lib"])[1] is True
    assert globals_for(["--db", "/tmp/a.db", "index", "lib"])[0] == Path("/tmp/a.db")


def test_search_requires_exactly_one_query_kind():
    with pytest.raises(SystemExit):
        parse(["search"])
    with pytest.raises(SystemExit):
        parse(["search", "-c", "a", "-t", "b"])


def test_search_accepts_each_query_kind():
    for flag, value in [("-c", "x"), ("-t", "x"), ("-p", "x")]:
        assert parse(["search", flag, value]).command == "search"
    assert parse(["search", "-e", "folder"]).examples == Path("folder")


def test_index_defaults():
    args = parse(["index", "lib"])
    assert args.video_samples == 8
    assert args.no_faces is False
    assert args.force is False


def test_organize_mode_is_symlink_by_default():
    """Originals must stay put unless the user asks otherwise."""
    assert parse(["search", "-c", "x", "-o", "out"]).mode == "symlink"


def test_unknown_mode_is_rejected():
    with pytest.raises(SystemExit):
        parse(["search", "-c", "x", "-o", "out", "--mode", "teleport"])


def test_forget_kind_is_constrained():
    assert parse(["forget", "concept", "x"]).kind == "concept"
    with pytest.raises(SystemExit):
        parse(["forget", "sideways", "x"])


def test_status_on_a_fresh_index(tmp_path, capsys):
    assert main(["--db", str(tmp_path / "i.db"), "status"]) == 0
    assert "files:      0" in capsys.readouterr().out


def test_listing_commands_are_friendly_when_empty(tmp_path, capsys):
    db = str(tmp_path / "i.db")
    assert main(["--db", db, "concepts"]) == 0
    assert "no concepts taught yet" in capsys.readouterr().out
    assert main(["--db", db, "people"]) == 0
    assert "no people registered yet" in capsys.readouterr().out


def test_forget_missing_concept_exits_nonzero(tmp_path, capsys):
    assert main(["--db", str(tmp_path / "i.db"), "forget", "concept", "ghost"]) == 1
    assert "no such concept" in capsys.readouterr().err


def test_search_on_empty_index_explains_itself(tmp_path, capsys):
    assert main(["--db", str(tmp_path / "i.db"), "search", "-c", "x"]) == 1
    assert "index is empty" in capsys.readouterr().err


def test_quiet_suppresses_progress_but_not_results(tmp_path, capsys):
    assert main(["--db", str(tmp_path / "i.db"), "-q", "concepts"]) == 0
    assert capsys.readouterr().out == ""
