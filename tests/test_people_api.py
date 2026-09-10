"""People, clustering, negatives, pending renames, and video thumbnails."""

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from siftr.server import create_app
from siftr.service import cluster_unnamed_faces, name_cluster
from siftr.vectors import normalize

TOKEN = "t"


@pytest.fixture
def client(tmp_path, embedder):
    app = create_app(db_path=tmp_path / "index.db", token=TOKEN)
    app.state.embedder = embedder
    app.state.allowlist.allow(tmp_path)
    with TestClient(app) as c:
        c.headers.update({"Authorization": f"Bearer {TOKEN}"})
        yield c


def _face(seed, dim=512):
    """A stand-in ArcFace embedding.

    512-d deliberately: in low dimensions random vectors are similar by chance,
    so a 16-d fixture would make any clustering threshold look broken. At
    ArcFace's real width unrelated faces are near-orthogonal, which is the
    regime the thresholds are actually tuned for.
    """
    rng = np.random.default_rng(seed)
    return normalize(rng.standard_normal(dim).astype(np.float32))


# -------------------------------------------------------------------- people


def test_people_starts_empty(client):
    assert client.get("/api/people").json()["people"] == []


def test_forget_unknown_person_is_404(client):
    assert client.delete("/api/people/nobody").status_code == 404


def test_person_files_for_unknown_person_is_empty(client):
    assert client.get("/api/people/ghost/files").json()["files"] == []


def test_rematch_with_no_people_is_zero(client):
    assert client.post("/api/people/rematch").json()["identified"] == 0


def test_person_files_lists_matches(client, tmp_path, db):
    """Exercised through the DB since faces need a model to detect."""
    from siftr.db import Database

    with Database(tmp_path / "index.db") as direct:
        person_id = direct.upsert_person("Nadia")
        face = _face(3)
        direct.replace_person_faces(person_id, face, ["ref.jpg"])
        file_id = direct.upsert_file(tmp_path / "a.jpg", "image", 1, 1)
        direct.add_file_faces(file_id, [{"vector": face, "person_id": person_id, "score": 0.9}])
        direct.commit()

    body = client.get("/api/people/Nadia/files").json()
    assert [f["name"] for f in body["files"]] == ["a.jpg"]
    assert client.get("/api/people").json()["people"][0]["files"] == 1


# ---------------------------------------------------------------- clustering


def test_clustering_groups_repeated_faces(db, tmp_path):
    """The 'who is this person appearing 40 times' case."""
    alice = _face(1)
    bob = _face(2)
    for i in range(6):
        file_id = db.upsert_file(tmp_path / f"{i}.jpg", "image", 1, i)
        # Small jitter so the faces are near-identical but not literally equal.
        vector = normalize(alice + _face(100 + i) * 0.05) if i < 4 else normalize(bob)
        db.add_file_faces(file_id, [{"vector": vector, "person_id": None, "score": None}])
    db.commit()

    clusters = cluster_unnamed_faces(db, threshold=0.5, min_size=3)

    assert clusters, "should find at least one group"
    assert clusters[0]["size"] == 4, "the four Alice faces should group"


def test_clustering_ignores_groups_below_min_size(db, tmp_path):
    for i in range(3):
        file_id = db.upsert_file(tmp_path / f"{i}.jpg", "image", 1, i)
        db.add_file_faces(file_id, [{"vector": _face(i), "person_id": None, "score": None}])
    db.commit()
    assert cluster_unnamed_faces(db, threshold=0.9, min_size=3) == []


def test_clustering_with_no_faces_is_empty(db):
    assert cluster_unnamed_faces(db) == []


def test_clustering_excludes_already_named_faces(db, tmp_path):
    person_id = db.upsert_person("Known")
    face = _face(7)
    for i in range(5):
        file_id = db.upsert_file(tmp_path / f"{i}.jpg", "image", 1, i)
        db.add_file_faces(file_id, [{"vector": face, "person_id": person_id, "score": 0.9}])
    db.commit()
    assert cluster_unnamed_faces(db) == []


def test_naming_a_cluster_creates_the_person_and_assigns_faces(db, tmp_path):
    alice = _face(1)
    for i in range(4):
        file_id = db.upsert_file(tmp_path / f"{i}.jpg", "image", 1, i)
        db.add_file_faces(
            file_id,
            [
                {
                    "vector": normalize(alice + _face(200 + i) * 0.05),
                    "person_id": None,
                    "score": None,
                }
            ],
        )
    db.commit()

    clusters = cluster_unnamed_faces(db, threshold=0.5, min_size=3)
    result = name_cluster(db, clusters[0]["face_ids"], "Alice")

    assert result.name == "Alice"
    assert len(db.files_for_person("Alice")) == 4
    assert cluster_unnamed_faces(db) == [], "named faces leave the unassigned pool"


def test_naming_an_empty_cluster_raises(db):
    with pytest.raises(ValueError):
        name_cluster(db, [], "Nobody")


def test_clusters_endpoint_reports_a_sample_path(client, tmp_path):
    from siftr.db import Database

    alice = _face(1)
    with Database(tmp_path / "index.db") as direct:
        for i in range(4):
            file_id = direct.upsert_file(tmp_path / f"{i}.jpg", "image", 1, i)
            direct.add_file_faces(
                file_id,
                [
                    {
                        "vector": normalize(alice + _face(300 + i) * 0.05),
                        "person_id": None,
                        "score": None,
                    }
                ],
            )
        direct.commit()

    body = client.get("/api/faces/clusters").json()
    assert body["clusters"][0]["size"] == 4
    assert body["clusters"][0]["sample_path"].endswith(".jpg")


# ----------------------------------------------------------------- negatives


def test_teach_accepts_negatives(client, tmp_path, make_images):
    make_images(tmp_path / "pos", (230, 20, 20), count=5, jitter=3)
    make_images(tmp_path / "neg", (20, 20, 230), count=5, jitter=3)
    body = client.post(
        "/api/tags",
        json={
            "name": "reds",
            "paths": [str(p) for p in (tmp_path / "pos").glob("*.png")],
            "negatives": [str(p) for p in (tmp_path / "neg").glob("*.png")],
        },
    ).json()
    assert body["name"] == "reds"
    assert 0 < body["threshold"] < 1


def test_negatives_outside_the_allowlist_are_refused(client, tmp_path, make_images):
    make_images(tmp_path / "pos", (230, 20, 20), count=3)
    outside = tmp_path.parent / "outside-neg"
    make_images(outside, (10, 10, 10), count=2)
    response = client.post(
        "/api/tags",
        json={
            "name": "x",
            "paths": [str(p) for p in (tmp_path / "pos").glob("*.png")],
            "negatives": [str(p) for p in outside.glob("*.png")],
        },
    )
    assert response.status_code == 403


# ----------------------------------------------------------- pending renames


def test_pending_is_zero_on_an_empty_library(client):
    assert client.get("/api/renames/pending").json()["pending"] == 0


def test_pending_counts_filenames_that_disagree_with_tags(client, tmp_path, make_images):
    from siftr.db import Database

    make_images(tmp_path / "lib", (10, 10, 10), count=1)
    target = tmp_path / "lib" / "img00.png"
    with Database(tmp_path / "index.db") as direct:
        concept_id = direct.save_concept("glaze", _face(1), 0.5, 2)
        file_id = direct.upsert_file(target, "image", 1, 1)
        direct.commit()
        direct.set_file_concepts(concept_id, [(file_id, 0.9)])

    body = client.get("/api/renames/pending").json()
    assert body["pending"] == 1
    assert body["changes"][0][1].endswith("img00 [glaze].png")


# ---------------------------------------------------------- video thumbnails


def test_thumb_decodes_a_video(client, tmp_path):
    """PIL cannot open a container; this used to 415 for every video."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")

    source = tmp_path / "src.png"
    from PIL import Image

    Image.new("RGB", (64, 64), (200, 40, 40)).save(source)
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(source),
            "-t",
            "1",
            "-r",
            "10",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )

    response = client.get("/api/thumb", params={"path": str(clip)})
    assert response.status_code == 200
    assert response.content[:2] == b"\xff\xd8"


# ------------------------------------------------------------ video samples


def test_raising_video_samples_invalidates_the_index(db, tmp_path):
    """A video embedded from 4 frames is not 'unchanged' when 8 are wanted."""
    path = tmp_path / "clip.mp4"
    file_id = db.upsert_file(path, "video", 100, 1, samples=4)
    db.add_embeddings(file_id, np.vstack([_face(i) for i in range(4)]), [0, 1, 2, 3])
    db.commit()

    assert db.is_unchanged(path, 100, 1, samples=4)
    assert not db.is_unchanged(path, 100, 1, samples=8), "must re-index at a higher rate"
    assert db.is_unchanged(path, 100, 1, samples=2), "lowering must not discard work"


# ------------------------------------------------- counter-examples only


def test_counter_examples_only_tightens_an_existing_tag(client, tmp_path, make_images):
    """ALT-dropping onto a tag says 'not this' without restating what it is."""
    make_images(tmp_path / "pos", (230, 20, 20), count=5, jitter=3)
    make_images(tmp_path / "neg", (20, 20, 230), count=4, jitter=3)
    positives = [str(p) for p in (tmp_path / "pos").glob("*.png")]
    negatives = [str(p) for p in (tmp_path / "neg").glob("*.png")]

    first = client.post("/api/tags", json={"name": "reds", "paths": positives}).json()

    tightened = client.post("/api/tags", json={"name": "reds", "paths": [], "negatives": negatives})
    assert tightened.status_code == 200, tightened.text
    body = tightened.json()
    assert body["examples"] == first["examples"], "the prototype's examples are unchanged"
    assert body["threshold"] >= first["threshold"], "the cutoff may only tighten or hold"


def test_counter_examples_for_an_unknown_tag_explain_themselves(client, tmp_path, make_images):
    make_images(tmp_path / "neg", (20, 20, 230), count=3)
    response = client.post(
        "/api/tags",
        json={
            "name": "never-taught",
            "paths": [],
            "negatives": [str(p) for p in (tmp_path / "neg").glob("*.png")],
        },
    )
    assert response.status_code == 400
    assert "does not exist yet" in response.json()["detail"]


def test_empty_teach_with_no_negatives_is_still_rejected(client):
    response = client.post("/api/tags", json={"name": "x", "paths": []})
    assert response.status_code == 400
