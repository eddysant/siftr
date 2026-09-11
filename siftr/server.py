"""Local HTTP API backing the desktop UI.

Why a server at all: loading CLIP costs several seconds of torch import plus
model init. Shelling out to the CLI per action would pay that on every click. A
resident process pays it once.

Security posture — this process can read any image on the machine and rename
files, so it is treated as privileged:

* Bound to 127.0.0.1 only, never a routable interface.
* Every request must carry a bearer token minted at startup. Without it any web
  page the user happens to have open could drive this API, since a browser will
  happily issue requests to localhost.
* Thumbnail and file reads are confined to directories the user has actually
  opened, the same allowlist idea photo-slap uses for its media:// protocol.
"""

from __future__ import annotations

import io
import os
import secrets
from pathlib import Path
from typing import Any

from .config import default_db_path
from .db import Database
from .faces import FaceRecognitionUnavailable
from .jobs import JobRunner


class Allowlist:
    """Directories the user has opened, and therefore consented to siftr reading.

    Paths are resolved before comparison so ``..`` cannot escape, and matching is
    done segment-wise so ``/photos-private`` is not treated as inside
    ``/photos``.
    """

    def __init__(self) -> None:
        self._roots: list[Path] = []

    def allow(self, path: Path) -> None:
        resolved = Path(path).expanduser().resolve()
        if resolved not in self._roots:
            self._roots.append(resolved)

    @property
    def roots(self) -> list[str]:
        return [str(r) for r in self._roots]

    def permits(self, path: Path) -> bool:
        try:
            resolved = Path(path).expanduser().resolve()
        except OSError:
            return False
        for root in self._roots:
            if resolved == root:
                return True
            # relative_to raises unless resolved is genuinely inside root, which
            # is the segment-wise check a string prefix would get wrong.
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def assert_permits(self, path: Path) -> Path:
        if not self.permits(path):
            raise PermissionError(f"path is outside every opened folder: {path}")
        return Path(path).expanduser().resolve()


# Request bodies live at module level, NOT inside create_app. With
# `from __future__ import annotations` every annotation is a string, and FastAPI
# resolves those against the function's module globals — a class defined inside
# create_app is invisible there, so FastAPI silently treats the parameter as a
# query scalar and every POST fails with a 422.
try:
    from pydantic import BaseModel

    class TeachBody(BaseModel):
        name: str
        paths: list[str]
        replace: bool = False
        # Explicit counter-examples. Calibration against the library makes
        # positives-only workable, but real negatives place the threshold more
        # accurately when the user can supply them.
        negatives: list[str] = []

    class PersonBody(BaseModel):
        name: str
        paths: list[str]
        all_faces: bool = False

    class NameClusterBody(BaseModel):
        name: str
        face_ids: list[int]

    class ReviewBody(BaseModel):
        path: str
        is_match: bool

    class SettingsBody(BaseModel):
        organize_mode: str

    class DestinationBody(BaseModel):
        destination: str | None = None
        #: Set the folder for files that do NOT match, rather than those that do.
        inverse: bool = False

    class OverrideBody(BaseModel):
        path: str
        tag: str
        state: str | None = None

    class IndexBody(BaseModel):
        folder: str
        video_samples: int = 8
        detect_faces: bool = True
        person_crops: bool = False
        force: bool = False
        workers: int | None = None

except ImportError:  # pragma: no cover - the UI extra is optional
    TeachBody = OverrideBody = IndexBody = None  # type: ignore[assignment]


def create_app(db_path: Path | None = None, token: str | None = None):
    """Build the FastAPI app. Imported lazily so the CLI need not depend on it."""
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Query
        from fastapi.responses import JSONResponse, Response
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The UI server needs extra dependencies. Install them with:\n"
            "    pip install 'siftr[ui]'"
        ) from exc

    resolved_db = Path(db_path) if db_path else default_db_path()
    api_token = token or os.environ.get("SIFTR_TOKEN") or secrets.token_urlsafe(32)

    app = FastAPI(title="siftr", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.token = api_token
    app.state.db_path = resolved_db
    app.state.allowlist = Allowlist()
    # Rebuild consent from the index: a library indexed in an earlier session or
    # from the CLI must be readable now, or every thumbnail 403s.
    with Database(resolved_db) as _seed:
        for root in _seed.list_roots():
            app.state.allowlist.allow(Path(root))
    app.state.jobs = JobRunner()
    app.state.embedder = None

    @app.exception_handler(PermissionError)
    def _denied(_request, exc: PermissionError):
        """Allowlist violations are a 403, not a 500.

        Registered app-wide so every path-touching endpoint gets the same
        answer and none can leak a stack trace by forgetting to catch it.
        """
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    def require_token(authorization: str = Header(default="")) -> None:
        expected = f"Bearer {app.state.token}"
        # compare_digest rather than == so a wrong token cannot be recovered by
        # timing the comparison.
        if not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid or missing token")

    guard = [Depends(require_token)]

    def open_db() -> Database:
        # A fresh connection per request: SQLite objects are not safe to share
        # across threads, and the job runner uses its own thread.
        return Database(resolved_db)

    def embedder():
        if app.state.embedder is None:
            from .embed import Embedder

            app.state.embedder = Embedder()
        return app.state.embedder

    # ------------------------------------------------------------- routes

    @app.get("/api/health")
    def health() -> dict:
        """Unauthenticated: lets the UI wait for the port without the token."""
        return {"ok": True, "model_loaded": app.state.embedder is not None}

    @app.get("/api/library", dependencies=guard)
    def library() -> dict:
        with open_db() as db:
            effective = db.effective_tags()
            files = []
            for row in db.library_view():
                files.append(
                    {
                        "id": int(row["id"]),
                        "path": row["path"],
                        "name": Path(row["path"]).name,
                        "kind": row["kind"],
                        "size": int(row["size"]),
                        "tags": effective.get(int(row["id"]), []),
                        "people": (row["people"] or "").split(",") if row["people"] else [],
                    }
                )
            return {"files": files, "roots": app.state.allowlist.roots}

    @app.get("/api/tags", dependencies=guard)
    def tags() -> dict:
        with open_db() as db:
            return {
                "tags": [
                    {
                        "name": row["name"],
                        "threshold": float(row["threshold"]),
                        "examples": int(row["n_examples"]),
                        "matches": int(row["n_matches"]),
                        "destination": row["destination"],
                        "inverse_destination": row["inverse_destination"],
                    }
                    for row in db.list_concepts()
                ]
            }

    @app.post("/api/tags", dependencies=guard)
    def teach(body: TeachBody) -> dict:
        """Teach a tag from dropped files. This is the drag-and-drop target."""
        from .service import example_paths_for, teach_from_paths

        paths = [app.state.allowlist.assert_permits(Path(p)) for p in body.paths]
        with open_db() as db:
            if not body.replace:
                # Dropping onto an existing tag adds to it rather than
                # redefining it, and pinned corrections count as examples too.
                paths = list(dict.fromkeys([*paths, *map(Path, example_paths_for(db, body.name))]))
            negatives = [app.state.allowlist.assert_permits(Path(p)) for p in body.negatives]
            try:
                result = teach_from_paths(db, body.name, paths, embedder(), negatives=negatives)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {
                "name": result.name,
                "examples": result.n_examples,
                "threshold": result.threshold,
                "cohesion": result.cohesion,
            }

    @app.get("/api/tags/{name}/boundary", dependencies=guard)
    def boundary(name: str, limit: int = Query(default=12)) -> dict:
        """Files the tag is least sure about — the ones worth confirming."""
        from .service import boundary_files

        with open_db() as db:
            try:
                files = boundary_files(db, name, limit)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            concept = db.get_concept(name)
            return {"tag": name, "threshold": float(concept["threshold"]), "files": files}

    @app.post("/api/tags/{name}/review", dependencies=guard)
    def review(name: str, body: ReviewBody) -> dict:
        """Answer yes/no about one boundary file, then re-learn the tag."""
        from .service import review_boundary_file

        path = app.state.allowlist.assert_permits(Path(body.path))
        with open_db() as db:
            try:
                result = review_boundary_file(db, name, path, body.is_match, embedder())
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {
                "tag": result.name,
                "threshold": result.threshold,
                "examples": result.n_examples,
                "rejections": len(db.rejections(int(db.get_concept(name)["id"]))),
            }

    @app.delete("/api/tags/{name}", dependencies=guard)
    def forget_tag(name: str) -> dict:
        with open_db() as db:
            if not db.delete_concept(name):
                raise HTTPException(status_code=404, detail=f"no such tag: {name}")
            return {"deleted": name}

    @app.post("/api/override", dependencies=guard)
    def override(body: OverrideBody) -> dict:
        """Force a tag on, suppress it, or hand the decision back to the model."""
        with open_db() as db:
            concept = db.get_concept(body.tag)
            if concept is None:
                raise HTTPException(status_code=404, detail=f"no such tag: {body.tag}")
            file_id = db.file_id_for_path(body.path)
            if file_id is None:
                raise HTTPException(status_code=404, detail="file is not indexed")
            try:
                db.set_override(file_id, int(concept["id"]), body.state)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"path": body.path, "tag": body.tag, "state": body.state}

    @app.get("/api/search", dependencies=guard)
    def search(
        tags: str = Query(default=""),
        mode: str = Query(default="any"),
        limit: int = Query(default=500),
    ) -> dict:
        names = [t for t in (t.strip() for t in tags.split(",")) if t]
        if mode not in {"any", "all"}:
            raise HTTPException(status_code=400, detail="mode must be 'any' or 'all'")
        with open_db() as db:
            rows = db.files_matching_tags(names, mode=mode, limit=limit)
            return {
                "mode": mode,
                "tags": names,
                "files": [
                    {
                        "id": int(r["id"]),
                        "path": r["path"],
                        "name": Path(r["path"]).name,
                        "kind": r["kind"],
                        "score": float(r["score"]),
                        "tags": (r["tags"] or "").split(","),
                    }
                    for r in rows
                ],
            }

    @app.post("/api/index", dependencies=guard)
    def start_index(body: IndexBody) -> dict:
        from .index import build_index

        folder = Path(body.folder).expanduser().resolve()
        if not folder.exists():
            raise HTTPException(status_code=404, detail=f"no such folder: {folder}")
        app.state.allowlist.allow(folder)
        with open_db() as db:
            db.add_root(folder)

        def work(job) -> dict:
            with open_db() as db:

                def progress(message: str) -> None:
                    job.message = message
                    job.current += 1

                stats = build_index(
                    db,
                    folder,
                    embedder(),
                    video_samples=body.video_samples,
                    detect_faces=body.detect_faces,
                    crop_people_regions=body.person_crops,
                    force=body.force,
                    workers=body.workers,
                    progress=progress,
                    should_stop=lambda: job.cancelled,
                )
                return {
                    "indexed": stats.indexed,
                    "regions": stats.regions_embedded,
                    "unchanged": stats.skipped_unchanged,
                    "failed": stats.failed,
                    "faces": stats.faces_found,
                    "errors": stats.errors[:50],
                }

        try:
            job = app.state.jobs.submit("index", work)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job.as_dict()

    @app.post("/api/score", dependencies=guard)
    def start_score(organize_after: bool = Query(default=True, alias="rename")) -> dict:
        """Score the library, then apply its organize policy.

        The policy — rename in place, file into folders, or neither — is a stored
        setting rather than a parameter, so the same choice applies however
        scoring was triggered.
        """
        from .service import organize, score_library

        def work(job) -> dict:
            with open_db() as db:
                score_library(db, job)
                out: dict[str, Any] = {"scored": True}
                if organize_after:
                    mode = db.get_setting("organize_mode", "rename")
                    job.message = (
                        "writing tags into filenames"
                        if mode == "rename"
                        else "filing into folders"
                        if mode == "move"
                        else "done"
                    )
                    roots = app.state.allowlist.roots
                    out["organize"] = organize(
                        db, mode=mode, root=Path(roots[0]) if roots else None
                    )
                return out

        try:
            job = app.state.jobs.submit("score", work)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job.as_dict()

    @app.post("/api/renames/undo", dependencies=guard)
    def undo() -> dict:
        from .service import undo_renames

        roots = app.state.allowlist.roots
        if not roots:
            raise HTTPException(status_code=400, detail="no library folder is open")
        with open_db() as db:
            return undo_renames(db, Path(roots[0]))

    # ----------------------------------------------------------------- people

    @app.get("/api/people", dependencies=guard)
    def people() -> dict:
        with open_db() as db:
            return {
                "people": [
                    {
                        "name": row["name"],
                        "references": int(row["n_examples"]),
                        "files": int(row["n_files"]),
                    }
                    for row in db.list_people()
                ]
            }

    @app.post("/api/people", dependencies=guard)
    def add_person(body: PersonBody) -> dict:
        """Register someone from dropped photos of them."""
        from .service import register_person_from_paths

        paths = [app.state.allowlist.assert_permits(Path(p)) for p in body.paths]
        with open_db() as db:
            try:
                result = register_person_from_paths(db, body.name, paths, body.all_faces)
            except FaceRecognitionUnavailable as exc:
                raise HTTPException(status_code=501, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {
                "name": result.name,
                "references": result.n_references,
                "skipped": result.skipped,
            }

    @app.delete("/api/people/{name}", dependencies=guard)
    def forget_person(name: str) -> dict:
        with open_db() as db:
            if not db.delete_person(name):
                raise HTTPException(status_code=404, detail=f"no such person: {name}")
            return {"deleted": name}

    @app.get("/api/people/{name}/files", dependencies=guard)
    def person_files(name: str, limit: int = Query(default=500)) -> dict:
        from .search import by_person

        with open_db() as db:
            hits = by_person(db, name, limit)
            return {
                "person": name,
                "files": [
                    {
                        "path": str(hit.path),
                        "name": hit.path.name,
                        "kind": hit.kind,
                        "score": hit.score,
                    }
                    for hit in hits
                ],
            }

    @app.post("/api/people/rematch", dependencies=guard)
    def rematch() -> dict:
        from .index import rematch_faces

        with open_db() as db:
            return {"identified": rematch_faces(db)}

    @app.get("/api/faces/clusters", dependencies=guard)
    def clusters(threshold: float = Query(default=0.5), min_size: int = Query(default=3)) -> dict:
        """Unidentified faces grouped into probable people, largest first."""
        from .service import cluster_unnamed_faces

        with open_db() as db:
            found = cluster_unnamed_faces(db, threshold=threshold, min_size=min_size)
            for cluster in found:
                cluster["sample_path"] = db.path_for(cluster["sample_file_id"])
            return {"clusters": found}

    @app.post("/api/faces/clusters/name", dependencies=guard)
    def name_a_cluster(body: NameClusterBody) -> dict:
        from .service import name_cluster

        with open_db() as db:
            try:
                result = name_cluster(db, body.face_ids, body.name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"name": result.name, "references": result.n_references}

    # --------------------------------------------------------------- settings

    @app.get("/api/settings", dependencies=guard)
    def settings() -> dict:
        with open_db() as db:
            return {"organize_mode": db.get_setting("organize_mode", "rename")}

    @app.put("/api/settings", dependencies=guard)
    def update_settings(body: SettingsBody) -> dict:
        from .service import ORGANIZE_MODES

        if body.organize_mode not in ORGANIZE_MODES:
            raise HTTPException(
                status_code=400, detail=f"organize_mode must be one of {ORGANIZE_MODES}"
            )
        with open_db() as db:
            db.set_setting("organize_mode", body.organize_mode)
            return {"organize_mode": body.organize_mode}

    @app.put("/api/tags/{name}/destination", dependencies=guard)
    def set_destination(name: str, body: DestinationBody) -> dict:
        """Where this tag's matches are filed in move mode."""
        if body.destination:
            # The destination is written to, so it needs the same consent as any
            # other folder siftr touches.
            app.state.allowlist.assert_permits(Path(body.destination))
        with open_db() as db:
            if not db.set_destination(name, body.destination, body.inverse):
                raise HTTPException(status_code=404, detail=f"no such tag: {name}")
            return {
                "name": name,
                "destination": body.destination,
                "inverse": body.inverse,
            }

    @app.post("/api/organize", dependencies=guard)
    def organize_now(dry_run: bool = Query(default=False)) -> dict:
        """Apply the organize policy without re-scoring."""
        from .service import organize

        with open_db() as db:
            roots = app.state.allowlist.roots
            return organize(db, root=Path(roots[0]) if roots else None, dry_run=dry_run)

    @app.post("/api/tags/{name}/verify", dependencies=guard)
    def verify(
        name: str,
        phrase: str | None = Query(default=None),
        candidates: int = Query(default=200),
    ) -> dict:
        """Second-pass verification. Slow by nature — runs as a job."""
        from .grounding import GroundingUnavailable
        from .service import verify_tag

        def work(job) -> dict:
            with open_db() as db:

                def progress(message: str) -> None:
                    job.message = message
                    job.current += 1

                try:
                    return {"results": verify_tag(db, name, phrase, candidates, progress)}
                except GroundingUnavailable as exc:
                    raise RuntimeError(str(exc)) from exc

        try:
            job = app.state.jobs.submit("verify", work)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job.as_dict()

    @app.get("/api/duplicates", dependencies=guard)
    def duplicates(distance: float | None = Query(default=None)) -> dict:
        """Duplicate and near-duplicate groups, largest first."""
        from .duplicates import DEFAULT_DISTANCE, find_duplicates

        cutoff = DEFAULT_DISTANCE if distance is None else distance
        with open_db() as db:
            groups = find_duplicates(db.duplicate_rows(), cutoff)
            return {
                "distance": cutoff,
                "groups": [
                    {
                        "kind": g.kind,
                        "distance": g.distance,
                        "keeper": str(g.keeper),
                        "paths": [str(p) for p in g.paths],
                        "redundant": [str(p) for p in g.redundant()],
                    }
                    for g in groups
                ],
                "redundant_total": sum(g.size - 1 for g in groups),
            }

    # -------------------------------------------------------------- filenames

    @app.get("/api/renames/pending", dependencies=guard)
    def pending_renames() -> dict:
        """How many filenames no longer match the tags their file carries.

        Undoing a rename batch restores names without reverting the tag changes
        that produced them, so the two can legitimately disagree. Reporting the
        count lets the UI say so instead of leaving it silent.
        """
        from .service import apply_tags_to_filenames

        with open_db() as db:
            preview = apply_tags_to_filenames(db, root=None, dry_run=True)
            return {"pending": preview["renamed"], "changes": preview["changes"][:50]}

    @app.get("/api/jobs/{job_id}", dependencies=guard)
    def job_status(job_id: str) -> dict:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job.as_dict()

    @app.post("/api/jobs/{job_id}/cancel", dependencies=guard)
    def job_cancel(job_id: str) -> dict:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        job.cancel()
        return job.as_dict()

    @app.get("/api/thumb", dependencies=guard)
    def thumb(path: str = Query(...), size: int = Query(default=320)) -> Response:
        """A downscaled JPEG for the grid.

        The renderer must never paint full-resolution originals — a 48MP photo is
        a ~200MB texture, and a grid holds hundreds at once.
        """
        from PIL import Image

        from .media import classify, poster_frame

        try:
            resolved = app.state.allowlist.assert_permits(Path(path))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

        # Videos need a decoded frame — PIL cannot open a container, so calling
        # load_image on an .mp4 used to 415 and every video in the grid showed a
        # broken image.
        kind = classify(resolved) or "image"
        try:
            image = poster_frame(resolved, kind)
        except Exception as exc:
            raise HTTPException(status_code=415, detail=f"cannot decode: {exc}") from exc

        image.thumbnail((size, size), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82)
        return Response(
            content=buffer.getvalue(),
            media_type="image/jpeg",
            # Keyed only on path+size, so a file edited in place would otherwise
            # keep serving a stale thumbnail.
            headers={"Cache-Control": "no-cache"},
        )

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, db_path: Path | None = None) -> None:
    """Run the API. Prints the token on stdout so a launcher can capture it."""
    import uvicorn

    app = create_app(db_path)
    # stdout is the handshake channel for the Electron launcher; it reads this
    # line to learn the token rather than sharing a file or env var.
    print(f"SIFTR_TOKEN={app.state.token}", flush=True)
    print(f"SIFTR_URL=http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
