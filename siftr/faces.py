"""Face detection and recognition for tagging the people you know.

Uses InsightFace (RetinaFace detection + ArcFace 512-d embeddings) rather than
``face_recognition``/dlib: it installs from a wheel on current Pythons, is
markedly more accurate across pose and age, and runs on ONNX Runtime without a
compiler toolchain.

You register a person by pointing siftr at a folder of photos of them. Every
reference embedding is kept rather than averaged, and matching takes the *best*
similarity across a person's references — a single centroid blurs together the
genuinely different appearances a face has at different ages and angles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .vectors import normalize

#: Cosine similarity above which two ArcFace embeddings are treated as the same
#: person. ArcFace is trained so that same-identity pairs cluster well above
#: ~0.28 and different identities below it; 0.38 is deliberately conservative,
#: preferring a missed tag over a wrong one.
DEFAULT_MATCH_THRESHOLD = 0.38

#: Detected faces smaller than this (pixels on the short side of the box) are
#: dropped. Tiny background faces produce unstable embeddings that generate
#: false matches far more often than useful tags.
MIN_FACE_PIXELS = 40


class FaceRecognitionUnavailable(RuntimeError):
    """Raised when the optional face dependencies are not installed."""


@dataclass
class DetectedFace:
    vector: np.ndarray
    bbox: tuple[int, int, int, int]
    det_score: float
    frame_time: float = 0.0

    @property
    def size(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return min(x2 - x1, y2 - y1)


class FaceAnalyzer:
    """Lazily-initialized InsightFace wrapper."""

    def __init__(self, model_name: str = "buffalo_l", det_size: int = 640):
        self.model_name = model_name
        self.det_size = det_size
        self._app = None

    def _ensure_app(self):
        if self._app is None:
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise FaceRecognitionUnavailable(
                    "Face recognition needs the optional extras. Install them with:\n"
                    "    pip install 'siftr[faces]'"
                ) from exc

            app = FaceAnalysis(
                name=self.model_name,
                # Only detection and recognition are needed; skipping the
                # landmark/age/gender models cuts load time and memory.
                allowed_modules=["detection", "recognition"],
            )
            app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
            self._app = app
        return self._app

    def detect(self, image: Image.Image, frame_time: float = 0.0) -> list[DetectedFace]:
        """Detect and embed every usable face in one image."""
        app = self._ensure_app()
        # InsightFace expects BGR, matching OpenCV's convention.
        array = np.asarray(image.convert("RGB"))[:, :, ::-1]
        faces = app.get(array)

        out: list[DetectedFace] = []
        for face in faces:
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = normalize(np.asarray(face.embedding, dtype=np.float32))
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            detected = DetectedFace(
                vector=np.asarray(embedding, dtype=np.float32),
                bbox=(x1, y1, x2, y2),
                det_score=float(getattr(face, "det_score", 0.0)),
                frame_time=frame_time,
            )
            if detected.size >= MIN_FACE_PIXELS:
                out.append(detected)
        return out


def largest_face(faces: list[DetectedFace]) -> DetectedFace | None:
    """The biggest face in a frame — the subject of a reference photo.

    Reference folders routinely contain group shots where the person you are
    registering is the one in front, so taking the largest face is a better
    default than taking all of them and polluting the reference set.
    """
    return max(faces, key=lambda f: f.size) if faces else None


def match(
    vector: np.ndarray,
    reference_matrix: np.ndarray,
    owners: list[tuple[int, str]],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> tuple[int, str, float] | None:
    """Match one face against all registered references.

    Returns ``(person_id, name, score)`` for the single best reference above the
    threshold, or ``None``. Comparing against every reference and taking the max
    is what lets several photos of the same person cover different appearances.
    """
    if reference_matrix.size == 0 or not owners:
        return None
    scores = normalize(np.atleast_2d(reference_matrix)) @ normalize(vector)
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    if best_score < threshold:
        return None
    person_id, name = owners[best_index]
    return person_id, name, best_score


def bbox_to_text(bbox: tuple[int, int, int, int]) -> str:
    return ",".join(str(int(v)) for v in bbox)
