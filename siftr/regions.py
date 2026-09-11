"""Person regions: embedding crops of people, not just whole frames.

CLIP embeds an entire image into one vector, so an attribute that occupies a
small part of the frame barely moves it. Measured: a feature covering 1% of the
frame shifts the embedding by 0.012 cosine and gives a prototype +0.008
separation over negatives; at 50% it gives +0.093. A tattooed forearm in a candid
photo is a few percent of the frame.

Cropping to the person should invert that ratio — a sleeve goes from ~3% of a
photo to a large share of the crop.

**Measured, it does not work, and this is off by default.** On 26 real
photographs (13 with visible arm tattoos, 13 without), ranking held-out positives
above negatives gave AUC 0.90 from whole frames and 0.80-0.82 from crops, across
five box geometries from head-and-shoulders to near-full-figure. Widening the box
to include arms did not recover it.

Two likely reasons, neither fixable by tuning the geometry. CLIP is trained on
whole images paired with captions, so a tight crop is out of distribution. And
per-file score is a max over that file's embeddings, so adding vectors raises the
*negative* ceiling as much as the positive one — mean negative score rose from
0.406 to 0.477.

The code is kept because the result is one attribute on one dataset, and an
attribute that is centred on a person rather than spread across them (a hat, say)
might behave differently. Turn it on with `siftr index --person-crops` and
measure before trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

#: Body box as multiples of the face box. A face is roughly one seventh of a
#: standing figure, but a crop that tall is mostly legs and background for a
#: seated or cropped subject, so this targets head-and-torso: the region that
#: carries clothing, arms and the attributes worth tagging.
WIDTH_FACES = 3.2
ABOVE_FACES = 0.6
BELOW_FACES = 4.5

#: Crops smaller than this on the short side are not embedded. A 40px person in
#: a crowd shot carries no attribute information, and embedding it only adds a
#: vector that can match something by accident.
MIN_CROP_PIXELS = 96


@dataclass(frozen=True)
class Region:
    """A crop of one person, with the box it came from."""

    image: Image.Image
    box: tuple[int, int, int, int]
    frame_time: float = 0.0


def body_box(face: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Expand a face box into a head-and-torso box, clamped to the image.

    Clamping rather than shifting: a face at the edge of the frame genuinely has
    less body visible, and sliding the box inward to preserve its size would
    crop in a neighbour instead.
    """
    x1, y1, x2, y2 = face
    width, height = size
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    centre = (x1 + x2) / 2

    half = face_w * WIDTH_FACES / 2
    left = int(max(0, centre - half))
    right = int(min(width, centre + half))
    top = int(max(0, y1 - face_h * ABOVE_FACES))
    bottom = int(min(height, y2 + face_h * BELOW_FACES))
    return left, top, right, bottom


def crop_people(
    image: Image.Image,
    faces: list[tuple[int, int, int, int]],
    frame_time: float = 0.0,
    min_pixels: int = MIN_CROP_PIXELS,
) -> list[Region]:
    """Crop one region per detected face, skipping ones too small to be useful."""
    regions: list[Region] = []
    for face in faces:
        box = body_box(face, image.size)
        left, top, right, bottom = box
        if min(right - left, bottom - top) < min_pixels:
            continue
        regions.append(Region(image=image.crop(box), box=box, frame_time=frame_time))
    return regions


def box_to_text(box: tuple[int, int, int, int]) -> str:
    return ",".join(str(int(v)) for v in box)
