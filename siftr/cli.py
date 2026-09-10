"""Command line interface for siftr."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import default_db_path
from .db import Database


def _global_flags() -> argparse.ArgumentParser:
    """Flags accepted either before or after the subcommand.

    ``siftr index ~/Pictures -q`` is the natural way to type this, so the global
    flags are attached to every subparser as well as the root parser. The
    Every copy uses ``SUPPRESS`` defaults and the real defaults are applied in
    :func:`main`. argparse's subparser action parses into a fresh namespace and
    then copies its attributes onto the parent's, so a normal default on either
    side would silently clobber a value given on the other.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--db",
        type=Path,
        default=argparse.SUPPRESS,
        help="index location (default: ~/.siftr/index.db)",
    )
    common.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="suppress progress output",
    )
    return common


def resolve_globals(args: argparse.Namespace) -> tuple[Path, bool]:
    """The effective ``--db`` and ``--quiet``, wherever they were given."""
    db = getattr(args, "db", None)
    return (db if db is not None else default_db_path()), getattr(args, "quiet", False)


def build_parser() -> argparse.ArgumentParser:
    common = _global_flags()
    parser = argparse.ArgumentParser(
        prog="siftr",
        parents=[common],
        description=(
            "Find media by example. Teach siftr a concept with a folder of pictures, "
            "then search your whole library for more of the same thing."
        ),
    )
    parser.add_argument("--version", action="version", version=f"siftr {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser(
        "index", parents=[common], help="scan a folder and embed everything in it"
    )
    p_index.add_argument("folder", type=Path)
    p_index.add_argument(
        "--video-samples", type=int, default=8, help="frames to sample per video (default: 8)"
    )
    p_index.add_argument("--no-faces", action="store_true", help="skip face detection")
    p_index.add_argument("--force", action="store_true", help="re-embed even unchanged files")
    p_index.add_argument(
        "--prune", action="store_true", help="also drop index entries for deleted files"
    )

    p_teach = sub.add_parser(
        "teach", parents=[common], help="learn a concept from a folder of examples"
    )
    p_teach.add_argument("name")
    p_teach.add_argument("folder", type=Path, help="folder of example images")
    p_teach.add_argument(
        "--negatives",
        type=Path,
        default=None,
        help="folder of counter-examples; makes the threshold much more accurate",
    )
    p_teach.add_argument(
        "--percentile",
        type=float,
        default=10.0,
        help="percentile of examples to admit when setting the threshold (default: 10)",
    )
    p_teach.add_argument(
        "--apply", action="store_true", help="tag matching files in the index right away"
    )

    p_search = sub.add_parser("search", parents=[common], help="search the index")
    group = p_search.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--concept", help="a concept you taught")
    group.add_argument("-e", "--examples", type=Path, help="folder of examples (no concept saved)")
    group.add_argument("-t", "--text", help="free-text prompt")
    group.add_argument("-p", "--person", help="a registered person")
    p_search.add_argument("-n", "--limit", type=int, default=25)
    p_search.add_argument("--threshold", type=float, default=None, help="override the cutoff")
    p_search.add_argument("--scores", action="store_true", help="show similarity scores")
    p_search.add_argument(
        "-o", "--output", type=Path, default=None, help="collect results into this folder"
    )
    p_search.add_argument(
        "--mode",
        choices=["symlink", "copy", "move"],
        default="symlink",
        help="how to collect results (default: symlink, which leaves originals alone)",
    )
    p_search.add_argument("--dry-run", action="store_true")

    p_apply = sub.add_parser("apply", parents=[common], help="tag indexed files with a concept")
    p_apply.add_argument("name")
    p_apply.add_argument("--threshold", type=float, default=None)

    p_person = sub.add_parser(
        "add-person", parents=[common], help="register someone from a folder of their photos"
    )
    p_person.add_argument("name")
    p_person.add_argument("folder", type=Path)
    p_person.add_argument(
        "--all-faces",
        action="store_true",
        help="use every face in each photo, not just the largest (for solo shots only)",
    )
    p_person.add_argument(
        "--rematch",
        action="store_true",
        help="re-check already-indexed faces against this person",
    )

    sub.add_parser("concepts", parents=[common], help="list taught concepts")
    sub.add_parser("people", parents=[common], help="list registered people")
    sub.add_parser("status", parents=[common], help="show index statistics")

    p_forget = sub.add_parser("forget", parents=[common], help="delete a concept or person")
    p_forget.add_argument("kind", choices=["concept", "person"])
    p_forget.add_argument("name")

    p_rematch = sub.add_parser(
        "rematch", parents=[common], help="re-check unidentified faces against all people"
    )
    p_rematch.add_argument("--threshold", type=float, default=None)

    p_serve = sub.add_parser(
        "serve", parents=[common], help="run the local API that backs the desktop UI"
    )
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Resolved here rather than via argparse defaults — see _global_flags().
    db_path, quiet = resolve_globals(args)

    def say(message: str) -> None:
        if not quiet:
            print(message)

    try:
        with Database(db_path) as db:
            return _dispatch(args, db, say)
    except KeyboardInterrupt:
        print("\ninterrupted; work completed so far is saved", file=sys.stderr)
        return 130
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args, db: Database, say) -> int:
    handlers = {
        "index": _cmd_index,
        "teach": _cmd_teach,
        "search": _cmd_search,
        "apply": _cmd_apply,
        "add-person": _cmd_add_person,
        "concepts": _cmd_concepts,
        "people": _cmd_people,
        "status": _cmd_status,
        "forget": _cmd_forget,
        "rematch": _cmd_rematch,
        "serve": _cmd_serve,
    }
    return handlers[args.command](args, db, say)


def _embedder(args):
    # Imported here so that `concepts`, `people`, `status` and `forget` never pay
    # the torch import.
    from .embed import Embedder

    return Embedder()


def _cmd_index(args, db: Database, say) -> int:
    from .index import build_index

    if args.prune:
        removed = db.forget_missing()
        say(f"pruned {removed} missing file(s)")

    # Record the folder so the desktop UI may read it later; indexing is the
    # act of consent.
    db.add_root(args.folder)

    stats = build_index(
        db,
        args.folder,
        _embedder(args),
        video_samples=args.video_samples,
        detect_faces=not args.no_faces,
        force=args.force,
        progress=say,
    )
    say(f"\n{stats.summary()}")
    for error in stats.errors[:10]:
        print(f"  ! {error}", file=sys.stderr)
    if len(stats.errors) > 10:
        print(f"  ! ...and {len(stats.errors) - 10} more", file=sys.stderr)
    return 0


def _cmd_teach(args, db: Database, say) -> int:
    from .concepts import learn

    concept = learn(
        args.name,
        args.folder,
        _embedder(args),
        negatives=args.negatives,
        percentile=args.percentile,
    )
    db.save_concept(concept.name, concept.prototype, concept.threshold, concept.n_examples)

    say(
        f"learned '{concept.name}' from {concept.n_examples} example(s)\n"
        f"  threshold: {concept.threshold:.3f}\n"
        f"  cohesion:  {concept.cohesion:.3f}"
    )
    if concept.cohesion < 0.70:
        say(
            "  note: these examples are not very similar to each other, so this "
            "concept will be vague.\n"
            "        Narrowing the example folder usually helps more than adding to it."
        )
    if args.negatives is None:
        say("  tip: pass --negatives with a folder of counter-examples for a tighter threshold")

    if args.apply:
        from .index import apply_concept

        count = apply_concept(db, concept.name)
        say(f"  tagged {count} file(s) in the index")
    return 0


def _cmd_search(args, db: Database, say) -> int:
    from . import search as search_mod

    if db.count_files() == 0:
        print("the index is empty — run 'siftr index <folder>' first", file=sys.stderr)
        return 1

    if args.concept:
        hits = search_mod.by_concept(db, args.concept, args.limit, args.threshold)
    elif args.examples:
        hits = search_mod.by_examples(
            db, args.examples, _embedder(args), args.limit, args.threshold
        )
    elif args.text:
        hits = search_mod.by_text(db, args.text, _embedder(args), args.limit, args.threshold)
    else:
        hits = search_mod.by_person(db, args.person, args.limit)

    if not hits:
        say("no matches")
        return 0

    for hit in hits:
        if args.scores:
            print(f"{hit.score:.3f}  {hit.path}")
        else:
            print(hit.path)

    if args.output:
        from .organize import place

        result = place([h.path for h in hits], args.output, mode=args.mode, dry_run=args.dry_run)
        verb = "would place" if args.dry_run else "placed"
        say(f"\n{verb} {result.placed} file(s) in {args.output} ({args.mode})")
        for error in result.errors:
            print(f"  ! {error}", file=sys.stderr)
    return 0


def _cmd_apply(args, db: Database, say) -> int:
    from .index import apply_concept

    count = apply_concept(db, args.name, args.threshold)
    say(f"tagged {count} file(s) with '{args.name}'")
    return 0


def _cmd_add_person(args, db: Database, say) -> int:
    import numpy as np

    from .concepts import example_images
    from .faces import FaceAnalyzer, largest_face
    from .media import load_image

    analyzer = FaceAnalyzer()
    vectors: list[np.ndarray] = []
    sources: list[str] = []
    no_face: list[str] = []

    for path in example_images(args.folder):
        try:
            found = analyzer.detect(load_image(path))
        except Exception as exc:
            no_face.append(f"{path.name}: {exc}")
            continue
        if not found:
            no_face.append(f"{path.name}: no face detected")
            continue
        chosen = found if args.all_faces else [largest_face(found)]
        for face in chosen:
            vectors.append(face.vector)
            sources.append(str(path))

    if not vectors:
        print(f"error: no faces found in {args.folder}", file=sys.stderr)
        for note in no_face[:10]:
            print(f"  ! {note}", file=sys.stderr)
        return 1

    person_id = db.upsert_person(args.name)
    db.replace_person_faces(person_id, np.vstack(vectors), sources)
    say(f"registered '{args.name}' with {len(vectors)} reference face(s)")
    for note in no_face[:5]:
        say(f"  skipped {note}")

    if args.rematch:
        from .index import rematch_faces

        assigned = rematch_faces(db)
        say(f"  matched {assigned} previously unidentified face(s)")
    return 0


def _cmd_concepts(args, db: Database, say) -> int:
    rows = db.list_concepts()
    if not rows:
        say("no concepts taught yet — try 'siftr teach <name> <folder-of-examples>'")
        return 0
    for row in rows:
        print(
            f"{row['name']:<24} threshold={row['threshold']:.3f}  "
            f"examples={row['n_examples']:<4} tagged={row['n_matches']}"
        )
    return 0


def _cmd_people(args, db: Database, say) -> int:
    rows = db.list_people()
    if not rows:
        say("no people registered yet — try 'siftr add-person <name> <folder-of-photos>'")
        return 0
    for row in rows:
        print(f"{row['name']:<24} references={row['n_examples']:<4} files={row['n_files']}")
    return 0


def _cmd_status(args, db: Database, say) -> int:
    files = db.count_files()
    embeddings = db.conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    faces = db.conn.execute("SELECT count(*) FROM file_faces").fetchone()[0]
    named = db.conn.execute(
        "SELECT count(*) FROM file_faces WHERE person_id IS NOT NULL"
    ).fetchone()[0]
    print(f"index:      {db.path}")
    print(f"files:      {files}")
    print(f"embeddings: {embeddings}")
    print(f"faces:      {faces} ({named} identified)")
    print(f"concepts:   {len(db.list_concepts())}")
    print(f"people:     {len(db.list_people())}")
    return 0


def _cmd_forget(args, db: Database, say) -> int:
    removed = (
        db.delete_concept(args.name) if args.kind == "concept" else db.delete_person(args.name)
    )
    if not removed:
        print(f"error: no such {args.kind}: {args.name}", file=sys.stderr)
        return 1
    say(f"forgot {args.kind} '{args.name}'")
    return 0


def _cmd_serve(args, db: Database, say) -> int:
    from .server import serve

    # The database is opened per-request inside the server; close this one so the
    # two do not hold separate write connections to the same file.
    db.close()
    say(f"siftr serving on http://{args.host}:{args.port}")
    serve(host=args.host, port=args.port, db_path=db.path)
    return 0


def _cmd_rematch(args, db: Database, say) -> int:
    from .index import rematch_faces

    assigned = rematch_faces(db, args.threshold)
    say(f"identified {assigned} face(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
