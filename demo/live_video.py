"""Background video grounding for demo/server.py: runs track_person.process_frame on a live source.

The worker plays one of several sources, and the page can switch between them. A file source loops
forever, and each pass starts a fresh tracker. A webcam source (an integer index) runs until the
camera stops. Files are played in real time: frames are processed at the file's own frame rate, and
frames are skipped when processing falls behind, so playback never slows down. The latest annotated
frame is kept as JPEG for the MJPEG stream. The query and the source can be changed at any time from
another thread.
"""
import logging
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO

from track_person import Settings, TrackState, encode_query, next_fps, open_source, process_frame

JPEG_QUALITY: int = 80

logger = logging.getLogger("live-video")


@dataclass(frozen=True)
class VideoSource:
    """A playable source: `path` is a file path or a webcam index, `id` is what the page sends back."""

    id: str
    title: str
    path: str


@dataclass(frozen=True)
class VideoStatus:
    """What the page shows next to the stream."""

    sources: list[VideoSource]
    source: str             # id of the source being played
    query: str
    query_version: int      # increments on every query change
    query_frames: int       # frames processed since the current query was set
    frame: int              # frames processed from the current source
    loop: int               # how many times the current file has restarted
    dropped: int            # frames skipped to stay real time on the current source
    fps: float              # processing rate, not playback rate
    people: int
    target: int | None
    target_score: float | None
    threshold: float
    error: str | None


class VideoWorkerFailed(RuntimeError):
    """The background video thread stopped with an error."""


class UnknownSource(ValueError):
    """A source id that the server was not started with."""


def make_source(path: str) -> VideoSource:
    if path.isdigit():
        return VideoSource(f"webcam-{path}", f"Webcam {path}", path)
    stem = Path(path).stem
    return VideoSource(stem, stem.replace("-", " ").replace("_", " ").capitalize(), path)


def make_sources(paths: list[str]) -> list[VideoSource]:
    """One source per path; ids must be unique because the page selects sources by id."""
    sources = [make_source(p) for p in paths]
    ids = [s.id for s in sources]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"video sources must have unique file names, duplicated: {dupes} in {paths}")
    missing = [s.path for s in sources if not s.path.isdigit() and not Path(s.path).is_file()]
    if missing:
        raise FileNotFoundError(f"video files not found: {missing}")
    return sources


def thumbnail(src: VideoSource, width: int, at: float) -> bytes:
    """JPEG of the frame at fraction `at` of a video file, resized to `width` pixels wide."""
    cap = open_source(src.path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * at))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read a thumbnail frame at {at:.0%} of {src.path} ({total} frames)")
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok_jpg, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok_jpg:
        raise RuntimeError(f"JPEG encoding failed for the thumbnail of {src.path}")
    return buf.tobytes()


class LiveGrounder:
    """Connector between the video sources and the web server; owns the processing thread."""

    def __init__(self, embedder: SentenceTransformer, model_lock: threading.Lock, sources: list[VideoSource],
                 detector_path: str, settings: Settings, query_text: str) -> None:
        if not sources:
            raise ValueError("LiveGrounder needs at least one video source")
        self._embedder = embedder
        self._model_lock = model_lock
        self._sources = {s.id: s for s in sources}
        self._detector_path = detector_path
        self._settings = settings
        self._cond = threading.Condition()
        self._query_text = query_text
        self._query = self._encode(query_text)
        self._wanted: str = sources[0].id
        self._jpeg: bytes = b""
        self._seq: int = 0
        self._status = VideoStatus(sources, sources[0].id, query_text, 0, 0, 0, 0, 0, 0.0, 0, None, None,
                                   settings.threshold, None)
        self._thread = threading.Thread(target=self._run_guarded, name="live-video", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def status(self) -> VideoStatus:
        with self._cond:
            return self._status

    def status_dict(self) -> dict[str, object]:
        return asdict(self.status())

    def set_query(self, text: str) -> VideoStatus:
        """Encode the new description and make the worker use it from its next frame."""
        emb = self._encode(text)
        with self._cond:
            self._query_text, self._query = text, emb
            self._status = replace(self._status, query=text, query_version=self._status.query_version + 1,
                                   query_frames=0, target=None, target_score=None)
            logger.info("video query changed", extra={"query": text, "version": self._status.query_version})
            return self._status

    def set_source(self, source_id: str) -> VideoStatus:
        """Make the worker switch to another source from its next frame."""
        if source_id not in self._sources:
            raise UnknownSource(f"unknown video source {source_id!r}, expected one of {sorted(self._sources)}")
        with self._cond:
            self._wanted = source_id
            self._status = replace(self._status, source=source_id, frame=0, loop=0, dropped=0, people=0,
                                   target=None, target_score=None)
            logger.info("video source changed", extra={"source": source_id})
            return self._status

    def wait_frame(self, after_seq: int, timeout: float) -> tuple[int, bytes]:
        """Block until a frame newer than `after_seq` exists; returns (seq, jpeg)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > after_seq or self._status.error is not None, timeout):
                raise TimeoutError(f"no new video frame within {timeout}s (last seq {after_seq})")
            if self._status.error is not None:
                raise VideoWorkerFailed(self._status.error)
            return self._seq, self._jpeg

    def _encode(self, text: str) -> np.ndarray:
        with self._model_lock:
            return encode_query(self._embedder, text)

    def _wanted_source(self) -> str:
        with self._cond:
            return self._wanted

    def _run_guarded(self) -> None:
        try:
            while True:
                self._play(self._sources[self._wanted_source()])
        except Exception as e:  # thread boundary: record the failure for the HTTP side, then re-raise
            logger.exception("video worker failed", extra={"source": self._wanted_source()})
            with self._cond:
                self._status = replace(self._status, error=f"{type(e).__name__}: {e}")
                self._cond.notify_all()
            raise

    def _play(self, src: VideoSource) -> None:
        """Play `src` (looping a file) until the page asks for a different source."""
        webcam = src.path.isdigit()
        frame_idx, fps, loop, dropped = 0, 0.0, 0, 0
        while True:
            cap = open_source(src.path)
            src_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
            detector = YOLO(self._detector_path)          # fresh tracker for every pass
            states: dict[int, TrackState] = {}
            t_pass = time.perf_counter()
            n = 0                                          # frames consumed from the file this pass
            while True:
                if self._wanted_source() != src.id:
                    cap.release()
                    return
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                t0 = time.perf_counter()
                with self._cond:
                    query, query_text = self._query, self._query_text
                with self._model_lock:
                    res = process_frame(self._embedder, detector, frame, frame_idx, states, query, query_text, fps,
                                        self._settings)
                states = res.states
                fps = next_fps(fps, time.perf_counter() - t0)
                ok_jpg, buf = cv2.imencode(".jpg", res.shown, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok_jpg:
                    raise RuntimeError(f"JPEG encoding failed for frame {frame_idx} of {src.path}")
                with self._cond:
                    if self._wanted == src.id:
                        self._jpeg, self._seq = buf.tobytes(), self._seq + 1
                        self._status = replace(
                            self._status, source=src.id, query_frames=self._status.query_frames + 1,
                            frame=frame_idx, loop=loop, dropped=dropped, fps=round(fps, 1), people=res.people,
                            target=res.target,
                            target_score=None if res.target_score is None else round(res.target_score, 3))
                        self._cond.notify_all()
                frame_idx += 1
                if webcam:
                    continue
                lag = time.perf_counter() - (t_pass + n / src_fps)   # >0: behind the file's own clock
                if lag < 0:
                    time.sleep(-lag)
                skip = int(lag * src_fps)
                for _ in range(skip):                                 # drop frames to stay real time
                    if not cap.grab():
                        break
                n, dropped = n + skip, dropped + skip
            cap.release()
            if webcam:
                raise ConnectionError(f"webcam {src.path} stopped delivering frames after frame {frame_idx}")
            if n == 0:
                raise ValueError(f"video {src.path} has no readable frames")
            loop += 1
            logger.info("video restarted", extra={"source": src.id, "loop": loop})
