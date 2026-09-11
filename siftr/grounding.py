"""Verifying candidates with open-vocabulary detection.

CLIP embeds a whole frame into one vector, which is why an attribute occupying a
few percent of a photo is hard for it. OWLv2 *localises*: asked for "a tattooed
arm" it looks for one and reports where. On a 26-photo benchmark it ranked
positives above negatives at AUC 1.000 against 0.944 for a CLIP few-shot
prototype.

It cannot replace the index. Measured at 0.59 images/s against CLIP's 58.9 — 99x
slower, or 23 hours against 14 minutes for 50,000 photos — and unlike an
embedding, which is computed once and answers any later question, detection must
re-run for every new query.

So it is a second stage: CLIP narrows the library to candidates from the index
that already exists, and this re-ranks those. A few hundred candidates is a few
minutes, which is affordable for a deliberate question.

Image-guided querying was tried first, since it would have fitted siftr's
teach-by-dropping model without a phrase. It does not work — scores saturate at
1.000 for every image and ranking comes out at AUC 0.205, worse than chance — so
verification takes a text phrase stored on the tag.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MODEL = "google/owlv2-base-patch16-ensemble"


class GroundingUnavailable(RuntimeError):
    """Raised when the optional grounding dependencies are not installed."""


@dataclass
class Verdict:
    path: Path
    score: float
    box: tuple[int, int, int, int] | None


class Verifier:
    """Lazily-loaded OWLv2 wrapper."""

    def __init__(self, model_name: str = MODEL):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def _ensure(self):
        if self._model is None:
            try:
                import torch
                from transformers import Owlv2ForObjectDetection, Owlv2Processor
            except ImportError as exc:
                raise GroundingUnavailable(
                    "Verification needs the optional grounding extras. Install them with:\n"
                    "    pip install 'siftr[grounding]'"
                ) from exc

            self._processor = Owlv2Processor.from_pretrained(self.model_name)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_name).eval()
            self._torch = torch
        return self._model

    def verify(self, path: Path, phrases: Sequence[str]) -> Verdict:
        """Best detection score for any of ``phrases`` in one image."""
        from .media import load_image

        self._ensure()
        torch = self._torch
        image = load_image(Path(path))

        inputs = self._processor(text=[list(phrases)], images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)

        # threshold=0.0 so the best box is always returned; the caller decides
        # what score is good enough, and a hard filter here would throw away the
        # ranking information that makes this worth running at all.
        results = self._processor.post_process_grounded_object_detection(
            outputs, threshold=0.0, target_sizes=torch.tensor([image.size[::-1]])
        )[0]

        if not len(results["scores"]):
            return Verdict(Path(path), 0.0, None)

        best = int(results["scores"].argmax())
        box = tuple(int(v) for v in results["boxes"][best].tolist())
        return Verdict(Path(path), float(results["scores"][best]), box)  # type: ignore[arg-type]


def verify_all(
    paths: Sequence[Path],
    phrases: Sequence[str],
    verifier: Verifier | None = None,
    progress=None,
) -> list[Verdict]:
    """Verify a candidate list, best first.

    Deliberately serial: OWLv2 is heavy enough that the batch is bounded by the
    caller keeping the candidate list short, not by parallelism here.
    """
    verifier = verifier or Verifier()
    out: list[Verdict] = []
    for i, path in enumerate(paths, start=1):
        if progress:
            progress(f"[{i}/{len(paths)}] verifying {Path(path).name}")
        try:
            out.append(verifier.verify(Path(path), phrases))
        except GroundingUnavailable:
            raise
        except Exception:
            # An unreadable candidate simply fails verification; it should not
            # take the rest of the batch down.
            out.append(Verdict(Path(path), 0.0, None))
    out.sort(key=lambda v: v.score, reverse=True)
    return out
