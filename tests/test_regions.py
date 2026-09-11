"""Person-region crops.

CLIP embeds a whole image into one vector, so an attribute covering a few percent
of a frame barely registers. Cropping to the person inverts that ratio. These
cover the geometry and that the crops actually reach the index.
"""

import numpy as np
import pytest
from PIL import Image

from siftr.faces import DetectedFace
from siftr.index import build_index
from siftr.regions import (
    ABOVE_FACES,
    BELOW_FACES,
    MIN_CROP_PIXELS,
    WIDTH_FACES,
    body_box,
    box_to_text,
    crop_people,
)
from siftr.vectors import normalize

# ------------------------------------------------------------------- geometry


def test_body_box_is_wider_and_taller_than_the_face():
    face = (400, 300, 500, 420)  # 100 x 120
    left, top, right, bottom = body_box(face, (1600, 1200))
    assert right - left == pytest.approx(100 * WIDTH_FACES, abs=2)
    assert bottom - top == pytest.approx(120 * (1 + ABOVE_FACES + BELOW_FACES), abs=3)


def test_body_box_is_centred_on_the_face():
    face = (400, 300, 500, 420)
    left, _, right, _ = body_box(face, (1600, 1200))
    assert (left + right) / 2 == pytest.approx(450, abs=2)


def test_body_box_extends_downward_not_upward():
    """Torso, not sky: most of the expansion is below the face."""
    face = (400, 300, 500, 420)
    _, top, _, bottom = body_box(face, (1600, 1200))
    assert 300 - top < bottom - 420


def test_body_box_is_clamped_to_the_image():
    face = (10, 10, 60, 70)
    left, top, right, bottom = body_box(face, (200, 200))
    assert left >= 0 and top >= 0 and right <= 200 and bottom <= 200


def test_a_face_at_the_edge_is_clamped_not_slid_inward():
    """Sliding the box to keep its size would crop in a neighbour instead."""
    face = (0, 100, 80, 200)
    left, _, right, _ = body_box(face, (1000, 1000))
    assert left == 0
    assert right < 80 * WIDTH_FACES, "box should be truncated, not shifted"


def test_a_tiny_face_yields_a_tiny_box():
    face = (500, 500, 508, 510)
    left, _top, right, _bottom = body_box(face, (2000, 2000))
    assert (right - left) < 40


# --------------------------------------------------------------------- crops


def test_crop_people_returns_one_region_per_face():
    image = Image.new("RGB", (1600, 1200), (128, 128, 128))
    faces = [(400, 300, 500, 420), (900, 320, 990, 430)]
    regions = crop_people(image, faces)
    assert len(regions) == 2
    for region, face in zip(regions, faces, strict=True):
        assert region.box == body_box(face, image.size)
        assert region.image.size == (
            region.box[2] - region.box[0],
            region.box[3] - region.box[1],
        )


def test_crops_too_small_to_carry_detail_are_skipped():
    """A 40px person in a crowd shot adds a vector that can only mislead."""
    image = Image.new("RGB", (2000, 2000))
    assert crop_people(image, [(500, 500, 508, 510)]) == []


def test_min_pixels_is_configurable():
    image = Image.new("RGB", (2000, 2000))
    assert len(crop_people(image, [(500, 500, 508, 510)], min_pixels=4)) == 1


def test_crops_carry_the_frame_time():
    image = Image.new("RGB", (1600, 1200))
    regions = crop_people(image, [(400, 300, 500, 420)], frame_time=7.5)
    assert regions[0].frame_time == 7.5


def test_no_faces_means_no_crops():
    assert crop_people(Image.new("RGB", (800, 600)), []) == []


def test_box_to_text_roundtrips():
    assert box_to_text((1, 2, 3, 4)) == "1,2,3,4"


def test_min_crop_default_is_sane():
    assert 32 <= MIN_CROP_PIXELS <= 256


# ------------------------------------------------------------- into the index


class _Analyzer:
    """Reports one large face in the middle of every frame."""

    def __init__(self, count=1):
        self.count = count

    def detect(self, image, frame_time=0.0):
        width, height = image.size
        out = []
        for i in range(self.count):
            x = width // 4 + i * 60
            out.append(
                DetectedFace(
                    normalize(np.random.default_rng(i).standard_normal(16).astype(np.float32)),
                    (x, height // 4, x + width // 6, height // 4 + height // 5),
                    0.9,
                    frame_time,
                )
            )
        return out


def test_person_crops_are_embedded_and_stored(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 150, 130), count=3, jitter=0, size=(900, 900))
    stats = build_index(
        db,
        tmp_path / "lib",
        embedder,
        detect_faces=True,
        face_analyzer=_Analyzer(),
        crop_people_regions=True,
        workers=2,
    )

    assert stats.frames_embedded == 3
    assert stats.regions_embedded == 3
    regions = db.conn.execute("SELECT count(*) FROM embeddings WHERE region = 'person'").fetchone()[
        0
    ]
    assert regions == 3


def test_region_rows_record_their_box(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 150, 130), count=1, size=(900, 900))
    build_index(
        db,
        tmp_path / "lib",
        embedder,
        detect_faces=True,
        face_analyzer=_Analyzer(),
        crop_people_regions=True,
    )
    row = db.conn.execute("SELECT box FROM embeddings WHERE region = 'person'").fetchone()
    assert row["box"] and len(row["box"].split(",")) == 4


def test_several_people_give_several_crops(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 150, 130), count=2, size=(900, 900))
    stats = build_index(
        db,
        tmp_path / "lib",
        embedder,
        detect_faces=True,
        face_analyzer=_Analyzer(count=3),
        crop_people_regions=True,
        workers=2,
    )
    assert stats.regions_embedded == 6


def test_crops_are_off_by_default(db, tmp_path, make_images, embedder):
    """Measured as worse than whole frames for attribute tags, so opt-in."""
    make_images(tmp_path / "lib", (200, 150, 130), count=3, size=(900, 900))
    stats = build_index(
        db, tmp_path / "lib", embedder, detect_faces=True, face_analyzer=_Analyzer()
    )
    assert stats.regions_embedded == 0
    assert stats.frames_embedded == 3


def test_crops_can_be_turned_off(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 150, 130), count=3, size=(900, 900))
    stats = build_index(
        db,
        tmp_path / "lib",
        embedder,
        detect_faces=True,
        face_analyzer=_Analyzer(),
        crop_people_regions=False,
    )
    assert stats.regions_embedded == 0
    assert stats.frames_embedded == 3


def test_small_images_produce_no_crops(db, tmp_path, make_images, embedder):
    """A thumbnail-sized photo cannot yield a person crop worth embedding."""
    make_images(tmp_path / "lib", (200, 150, 130), count=2, size=(64, 64))
    stats = build_index(
        db,
        tmp_path / "lib",
        embedder,
        detect_faces=True,
        face_analyzer=_Analyzer(),
        crop_people_regions=True,
    )
    assert stats.faces_found == 2
    assert stats.regions_embedded == 0


def test_no_faces_means_no_region_embeddings(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 150, 130), count=3, size=(900, 900))
    stats = build_index(db, tmp_path / "lib", embedder, detect_faces=False)
    assert stats.regions_embedded == 0


def test_region_embeddings_participate_in_search(db, tmp_path, make_images, embedder):
    """The point of the whole feature: scoring takes the best over a file's
    embeddings, so a crop can match on its own terms."""
    from siftr import search as search_mod

    make_images(tmp_path / "lib", (200, 150, 130), count=2, size=(900, 900))
    build_index(
        db,
        tmp_path / "lib",
        embedder,
        detect_faces=True,
        face_analyzer=_Analyzer(),
        crop_people_regions=True,
    )

    total = db.conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    assert total == 4, "two frames plus two crops"

    make_images(tmp_path / "q", (200, 150, 130), count=1, size=(900, 900))
    hits = search_mod.by_examples(db, tmp_path / "q", embedder, limit=10)
    assert len(hits) == 2, "still one hit per file, not one per embedding"
