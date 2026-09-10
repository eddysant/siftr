"""HTTP API behaviour, including its security properties.

The real embedder is replaced with the deterministic fake, so these run without
downloading CLIP.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from siftr.server import Allowlist, create_app  # noqa: E402

TOKEN = "test-token"


@pytest.fixture
def client(tmp_path, embedder):
    app = create_app(db_path=tmp_path / "index.db", token=TOKEN)
    app.state.embedder = embedder
    app.state.allowlist.allow(tmp_path)
    with TestClient(app) as c:
        c.headers.update({"Authorization": f"Bearer {TOKEN}"})
        yield c


# ------------------------------------------------------------------ allowlist


def test_allowlist_permits_a_file_inside_an_opened_root(tmp_path):
    allow = Allowlist()
    allow.allow(tmp_path)
    assert allow.permits(tmp_path / "sub" / "a.jpg")


def test_allowlist_rejects_anything_outside(tmp_path):
    allow = Allowlist()
    allow.allow(tmp_path / "opened")
    assert not allow.permits(tmp_path / "elsewhere" / "a.jpg")


def test_allowlist_rejects_a_sibling_sharing_a_name_prefix(tmp_path):
    """/photos-private must not be treated as inside /photos."""
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos-private").mkdir()
    allow = Allowlist()
    allow.allow(tmp_path / "photos")
    assert not allow.permits(tmp_path / "photos-private" / "secret.jpg")


def test_allowlist_cannot_be_escaped_with_dotdot(tmp_path):
    (tmp_path / "opened").mkdir()
    (tmp_path / "secret").mkdir()
    allow = Allowlist()
    allow.allow(tmp_path / "opened")
    assert not allow.permits(tmp_path / "opened" / ".." / "secret" / "a.jpg")


def test_allowlist_empty_permits_nothing(tmp_path):
    assert not Allowlist().permits(tmp_path / "a.jpg")


# ----------------------------------------------------------------------- auth


def test_health_needs_no_token(tmp_path):
    app = create_app(db_path=tmp_path / "i.db", token=TOKEN)
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200


def test_endpoints_reject_a_missing_token(tmp_path):
    app = create_app(db_path=tmp_path / "i.db", token=TOKEN)
    with TestClient(app) as c:
        assert c.get("/api/library").status_code == 401
        assert c.get("/api/tags").status_code == 401
        assert c.post("/api/score").status_code == 401


def test_endpoints_reject_a_wrong_token(tmp_path):
    app = create_app(db_path=tmp_path / "i.db", token=TOKEN)
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer wrong"})
        assert c.get("/api/library").status_code == 401


def test_a_generated_token_is_not_guessable(tmp_path):
    a = create_app(db_path=tmp_path / "a.db")
    b = create_app(db_path=tmp_path / "b.db")
    assert a.state.token != b.state.token
    assert len(a.state.token) >= 32


# -------------------------------------------------------------------- library


def test_library_is_empty_before_indexing(client):
    body = client.get("/api/library").json()
    assert body["files"] == []


def test_tags_start_empty(client):
    assert client.get("/api/tags").json()["tags"] == []


# ---------------------------------------------------------------------- teach


def test_teach_from_dropped_files(client, tmp_path, make_images):
    make_images(tmp_path / "examples", (220, 30, 30), count=4)
    paths = [str(p) for p in sorted((tmp_path / "examples").glob("*.png"))]

    body = client.post("/api/tags", json={"name": "reds", "paths": paths}).json()

    assert body["name"] == "reds"
    assert body["examples"] == 4
    assert 0 <= body["threshold"] <= 1
    assert client.get("/api/tags").json()["tags"][0]["name"] == "reds"


def test_teach_rejects_paths_outside_the_allowlist(client, tmp_path, make_images):
    outside = tmp_path.parent / "outside-the-library"
    make_images(outside, (10, 10, 10), count=1)
    response = client.post("/api/tags", json={"name": "x", "paths": [str(outside / "img00.png")]})
    assert response.status_code == 403


def test_teach_rejects_an_invalid_tag_name(client, tmp_path, make_images):
    make_images(tmp_path / "ex", (10, 10, 10), count=2)
    paths = [str(p) for p in (tmp_path / "ex").glob("*.png")]
    assert client.post("/api/tags", json={"name": "bad[name]", "paths": paths}).status_code == 400


def test_teach_with_unreadable_files_is_a_400(client, tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    assert client.post("/api/tags", json={"name": "x", "paths": [str(broken)]}).status_code == 400


def test_forget_an_unknown_tag_is_404(client):
    assert client.delete("/api/tags/ghost").status_code == 404


def test_forget_removes_the_tag(client, tmp_path, make_images):
    make_images(tmp_path / "ex", (10, 200, 10), count=3)
    paths = [str(p) for p in (tmp_path / "ex").glob("*.png")]
    client.post("/api/tags", json={"name": "greens", "paths": paths})
    assert client.delete("/api/tags/greens").status_code == 200
    assert client.get("/api/tags").json()["tags"] == []


# --------------------------------------------------------------------- search


def test_search_rejects_a_bad_mode(client):
    assert client.get("/api/search", params={"tags": "a", "mode": "sideways"}).status_code == 400


def test_search_with_no_tags_returns_nothing(client):
    assert client.get("/api/search", params={"tags": ""}).json()["files"] == []


# ------------------------------------------------------------------ overrides


def test_override_on_an_unknown_tag_is_404(client, tmp_path):
    response = client.post(
        "/api/override", json={"path": str(tmp_path / "a.jpg"), "tag": "ghost", "state": "on"}
    )
    assert response.status_code == 404


# ----------------------------------------------------------------------- jobs


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404


def test_index_a_missing_folder_is_404(client, tmp_path):
    assert client.post("/api/index", json={"folder": str(tmp_path / "nope")}).status_code == 404


def test_thumb_rejects_a_path_outside_the_allowlist(client, tmp_path):
    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(b"x")
    assert client.get("/api/thumb", params={"path": str(outside)}).status_code == 403


def test_thumb_serves_a_jpeg_for_an_allowed_image(client, tmp_path, make_images):
    make_images(tmp_path / "lib", (10, 10, 200), count=1)
    response = client.get("/api/thumb", params={"path": str(tmp_path / "lib" / "img00.png")})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_thumb_on_an_undecodable_file_is_415(client, tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"nope")
    assert client.get("/api/thumb", params={"path": str(broken)}).status_code == 415
