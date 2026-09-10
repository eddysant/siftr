"""Finding media files and turning them into still frames.

Images contribute one frame. Videos are *sampled*: siftr embeds a handful of
frames spread across the clip rather than every frame, because a concept only
has to appear somewhere in a video for the file to be a match, and embedding
every frame of a long clip costs orders of magnitude more for almost no recall.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

# Register the HEIF/HEIC opener if it is available. Pillow ships no HEIC support
# of its own (HEVC is patent-encumbered), and a Mac photo library is mostly HEIC,
# so without this every iPhone photo fails to decode and is skipped at index time.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:  # pragma: no cover - optional dependency
    HEIC_SUPPORTED = False

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".wmv",
    ".flv",
    ".mpg",
    ".mpeg",
    ".m2ts",
}


@dataclass(frozen=True)
class MediaFile:
    path: Path
    kind: str  # 'image' | 'video'
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class Frame:
    """One still to embed, tagged with where in the file it came from."""

    image: Image.Image
    time: float


def classify(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def discover(root: Path, recursive: bool = True) -> Iterator[MediaFile]:
    """Yield every supported media file under ``root``.

    Hidden files and directories are skipped, as are macOS package directories
    (``.photoslibrary``, ``.app`` and friends) — walking into an Apple Photos
    library would index thousands of internal thumbnail derivatives alongside
    the real masters.
    """
    root = Path(root).expanduser()
    if root.is_file():
        kind = classify(root)
        if kind:
            stat = root.stat()
            yield MediaFile(root, kind, stat.st_size, stat.st_mtime_ns)
        return

    paths = root.rglob("*") if recursive else root.glob("*")
    for path in paths:
        if any(part.startswith(".") for part in path.parts):
            continue
        if any(_is_package(part) for part in path.parts[:-1]):
            continue
        if not path.is_file():
            continue
        kind = classify(path)
        if not kind:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        yield MediaFile(path, kind, stat.st_size, stat.st_mtime_ns)


_PACKAGE_SUFFIXES = (".photoslibrary", ".app", ".fcpbundle", ".aplibrary", ".imovielibrary")


def _is_package(name: str) -> bool:
    return name.lower().endswith(_PACKAGE_SUFFIXES)


def load_image(path: Path) -> Image.Image:
    """Open an image as RGB, honoring the EXIF orientation tag.

    Without ``exif_transpose`` a phone photo taken in portrait embeds sideways,
    which measurably degrades both face detection and concept similarity.
    """
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def poster_frame(path: Path, kind: str) -> Image.Image:
    """One representative still for a file, for thumbnails.

    Videos need a decoded frame; PIL cannot open a container. Sampling a single
    frame from a little way in avoids the black or fading first frame that many
    clips open on.
    """
    if kind == "image":
        return load_image(path)

    sampled = sample_video(path, samples=1)
    if not sampled:
        raise ValueError(f"no decodable frames in {path.name}")
    return sampled[0].image


def frames(path: Path, kind: str, samples: int = 8) -> list[Frame]:
    """Extract frames to embed from one media file."""
    if kind == "image":
        return [Frame(load_image(path), 0.0)]
    return sample_video(path, samples)


def sample_video(path: Path, samples: int = 8) -> list[Frame]:
    """Sample ``samples`` frames spread evenly through a video.

    Uses PyAV when installed (no subprocess, and seeking is precise); otherwise
    shells out to ffmpeg. Returns an empty list if neither is available or the
    file cannot be decoded — an unreadable video is skipped, never fatal.
    """
    try:
        return _sample_with_av(path, samples)
    except ImportError:
        return _sample_with_ffmpeg(path, samples)
    except Exception:
        return _sample_with_ffmpeg(path, samples)


def _sample_with_av(path: Path, samples: int) -> list[Frame]:
    import av  # optional dependency, imported lazily

    out: list[Frame] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            return []
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration = _av_duration(container, stream)

        if not duration:
            # Unknown duration (some streams report none): fall back to taking
            # the first frames we can decode rather than guessing seek targets.
            for i, frame in enumerate(container.decode(stream)):
                if i >= samples:
                    break
                out.append(Frame(frame.to_image().convert("RGB"), float(frame.time or 0.0)))
            return out

        # Skip the extreme head and tail: many clips open on black or a fade.
        for target in _sample_times(duration, samples):
            try:
                container.seek(int(target / stream.time_base), stream=stream)
                frame = next(container.decode(stream))
            except (StopIteration, av.AVError):
                continue
            out.append(Frame(frame.to_image().convert("RGB"), float(frame.time or target)))
    return out


def _av_duration(container, stream) -> float | None:
    if stream.duration and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration:
        import av

        return float(container.duration / av.time_base)
    return None


def _sample_times(duration: float, samples: int) -> list[float]:
    """Evenly spaced sample points, inset from both ends.

    The inset matters: sampling at t=0 very often lands on a black first frame,
    and at t=duration on a fade-out or past the last decodable packet.
    """
    if samples <= 1 or duration <= 0:
        return [max(0.0, duration / 2)]
    span = duration * 0.9
    start = duration * 0.05
    return [start + span * i / (samples - 1) for i in range(samples)]


def _sample_with_ffmpeg(path: Path, samples: int) -> list[Frame]:
    duration = _ffprobe_duration(path)
    if duration is None:
        return []
    out: list[Frame] = []
    for target in _sample_times(duration, samples):
        png = _ffmpeg_frame(path, target)
        if png is None:
            continue
        import io

        try:
            image = Image.open(io.BytesIO(png)).convert("RGB")
        except Exception:
            continue
        out.append(Frame(image, target))
    return out


def _ffprobe_duration(path: Path) -> float | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _ffmpeg_frame(path: Path, timestamp: float) -> bytes | None:
    try:
        proc = subprocess.run(
            [
                # -ss before -i seeks by keyframe, which is far faster on long
                # files and accurate enough for concept sampling.
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return proc.stdout or None
