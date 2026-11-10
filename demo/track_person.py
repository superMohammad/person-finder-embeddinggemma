"""Real-time text-guided person grounding in video: draw a box on the person who matches a description.

Per frame, YOLO finds people and ByteTrack keeps a stable id for each. Every `--refresh` frames a
track's crop is re-encoded by the EmbeddingGemma image tower (batched, GPU). The crop embeddings of
each track are smoothed with an exponential moving average. The track with the highest cosine
similarity to the text query is boxed in green when its score clears `--threshold`, and every
other person gets a thin grey box with their score.

Type a new description in the terminal and press Enter to change the query while the video runs.
Press q in the video window to quit.

Run from the project root:

  .venv/bin/python demo/track_person.py --model models/embeddinggemma2-rstpreid \\
      --source path/to/video.mp4 --query "a woman in a red coat carrying a black bag"

  --source 0 uses the first webcam; --output out.mp4 also writes the annotated video;
  --no-display runs headless (needs --output).
"""
import argparse
import logging
import queue
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO

QUERY_PROMPT: str = "SearchQuery"
PERSON_CLASS: int = 0
TRACKER_CFG: str = "bytetrack.yaml"
WINDOW: str = "track_person"
GREEN: tuple[int, int, int] = (134, 191, 63)      # BGR of #3fbf86, the match colour of the web pages
PAPER: tuple[int, int, int] = (232, 240, 243)     # BGR of the off-white paper #f3f0e8
BLACK: tuple[int, int, int] = (0, 0, 0)

logger = logging.getLogger("track-person")


@dataclass(frozen=True)
class Detection:
    """One tracked person in one frame; box is (x1, y1, x2, y2) in pixels."""

    track_id: int
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class TrackState:
    """Smoothed appearance of one track and when it was last re-encoded / seen."""

    embedding: np.ndarray   # (D,) L2-normalized float32
    last_encoded: int
    last_seen: int


@dataclass(frozen=True)
class Settings:
    refresh: int            # re-encode a track every this many frames
    ema: float              # weight of the newest crop embedding in the running average
    threshold: float        # minimum cosine score to box a track as the match
    pad: float              # fraction of box size added around each crop
    forget: int             # drop a track this many frames after it was last seen
    min_aspect: float       # box height / width a track needs to be eligible as the match
    conf: float             # person detection confidence
    imgsz: int              # detector input size
    batch: int              # crops per embedding batch


@dataclass(frozen=True)
class FrameResult:
    """Annotated frame plus the track table to carry into the next frame."""

    shown: np.ndarray
    states: dict[int, TrackState]
    people: int
    encoded: int
    target: int | None
    target_score: float | None


def load_embedder(model_id: str, vision_tokens: int) -> SentenceTransformer:
    model = SentenceTransformer(model_id, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16},
                                config_kwargs={"audio_config": None})
    model[0].processor.image_processor.max_soft_tokens = vision_tokens
    return model


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    return model.encode([query], prompt_name=QUERY_PROMPT, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)


def encode_crops(model: SentenceTransformer, crops: list[Image.Image], batch_size: int) -> np.ndarray:
    return model.encode(crops, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False).astype(np.float32)


def open_source(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video source {source!r}")
    return cap


def read_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


def detect_people(detector: YOLO, frame: np.ndarray, conf: float, imgsz: int) -> list[Detection]:
    result = detector.track(frame, persist=True, classes=[PERSON_CLASS], conf=conf, imgsz=imgsz,
                            tracker=TRACKER_CFG, verbose=False)[0]
    if result.boxes is None or result.boxes.id is None:
        return []
    ids: list[int] = result.boxes.id.int().tolist()
    boxes: list[list[int]] = result.boxes.xyxy.round().int().tolist()
    return [Detection(i, (b[0], b[1], b[2], b[3])) for i, b in zip(ids, boxes)]


def crop_person(frame: np.ndarray, box: tuple[int, int, int, int], pad: float) -> Image.Image:
    """Padded RGB crop, matching the loose pedestrian boxes of RSTPReid."""
    x1, y1, x2, y2 = box
    dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
    h, w = frame.shape[:2]
    patch = frame[max(0, y1 - dy):min(h, y2 + dy), max(0, x1 - dx):min(w, x2 + dx)]
    return Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))


def stale_detections(dets: list[Detection], states: dict[int, TrackState], frame_idx: int,
                     refresh: int) -> list[Detection]:
    return [d for d in dets if d.track_id not in states
            or frame_idx - states[d.track_id].last_encoded >= refresh]


def blend(old: np.ndarray, new: np.ndarray, ema: float) -> np.ndarray:
    mixed = (1.0 - ema) * old + ema * new
    return (mixed / np.linalg.norm(mixed)).astype(np.float32)


def update_states(states: dict[int, TrackState], dets: list[Detection], encoded: list[Detection],
                  embeddings: np.ndarray, frame_idx: int, s: Settings) -> dict[int, TrackState]:
    """New track table: fold fresh embeddings in, mark seen tracks, forget long-gone ones."""
    out: dict[int, TrackState] = {k: v for k, v in states.items() if frame_idx - v.last_seen <= s.forget}
    for d, e in zip(encoded, embeddings):
        prev = out.get(d.track_id)
        emb = e if prev is None else blend(prev.embedding, e, s.ema)
        out[d.track_id] = TrackState(emb, frame_idx, frame_idx)
    for d in dets:
        out[d.track_id] = replace(out[d.track_id], last_seen=frame_idx)
    return out


def score_tracks(dets: list[Detection], states: dict[int, TrackState], query: np.ndarray) -> dict[int, float]:
    return {d.track_id: float(states[d.track_id].embedding @ query) for d in dets}


def full_body_ids(dets: list[Detection], min_aspect: float) -> set[int]:
    """Tracks whose box is person-shaped (taller than wide), like the full-body RSTPReid crops.

    Head-only boxes at the frame edge and boxes merging two people score unreliably, so they are
    drawn and scored but never chosen as the match.
    """
    return {d.track_id for d in dets if (d.box[3] - d.box[1]) >= min_aspect * (d.box[2] - d.box[0])}


def best_match(scores: dict[int, float], eligible: set[int], threshold: float) -> int | None:
    candidates = [k for k in scores if k in eligible]
    if not candidates:
        return None
    tid = max(candidates, key=lambda k: scores[k])
    return tid if scores[tid] >= threshold else None


def draw(frame: np.ndarray, dets: list[Detection], scores: dict[int, float], target: int | None,
         query: str, fps: float) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d.box
        hit = d.track_id == target
        color, thick = (GREEN, 3) if hit else (PAPER, 1)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = f"{'MATCH ' if hit else ''}{scores[d.track_id]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty = max(th + 4, y1)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
        cv2.putText(out, label, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLACK, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, f"{fps:4.1f} fps | {query}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def process_frame(embedder: SentenceTransformer, detector: YOLO, frame: np.ndarray, frame_idx: int,
                  states: dict[int, TrackState], query: np.ndarray, query_text: str, fps: float,
                  s: Settings) -> FrameResult:
    """Detect and track people, refresh stale crop embeddings, score them against the query, draw."""
    dets = detect_people(detector, frame, s.conf, s.imgsz)
    todo = stale_detections(dets, states, frame_idx, s.refresh)
    embs = (encode_crops(embedder, [crop_person(frame, d.box, s.pad) for d in todo], s.batch)
            if todo else np.zeros((0, query.shape[0]), dtype=np.float32))
    new_states = update_states(states, dets, todo, embs, frame_idx, s)
    scores = score_tracks(dets, new_states, query)
    target = best_match(scores, full_body_ids(dets, s.min_aspect), s.threshold)
    return FrameResult(draw(frame, dets, scores, target, query_text, fps), new_states, len(dets), len(todo),
                       target, None if target is None else scores[target])


def add_tracking_args(p: argparse.ArgumentParser) -> None:
    """Detector and tracking options shared by this CLI and demo/server.py."""
    p.add_argument("--detector", required=False, default="models/yolo26s.pt", help="Ultralytics YOLO weights")
    p.add_argument("--vision-tokens", type=int, required=False, default=70, choices=[70, 140, 280, 560, 1120])
    p.add_argument("--refresh", type=int, required=False, default=10, help="re-encode each track every N frames")
    p.add_argument("--ema", type=float, required=False, default=0.3)
    p.add_argument("--threshold", type=float, required=False, default=0.45)
    p.add_argument("--pad", type=float, required=False, default=0.05)
    p.add_argument("--forget", type=int, required=False, default=60)
    p.add_argument("--min-aspect", type=float, required=False, default=1.2,
                   help="box height / width needed to be chosen as the match (rejects head-only and merged boxes)")
    p.add_argument("--conf", type=float, required=False, default=0.4, help="person detection confidence")
    p.add_argument("--imgsz", type=int, required=False, default=640)
    p.add_argument("--batch", type=int, required=False, default=32)


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(args.refresh, args.ema, args.threshold, args.pad, args.forget, args.min_aspect, args.conf, args.imgsz,
                    args.batch)


def next_fps(fps: float, seconds: float) -> float:
    """Exponential moving average of the processing rate."""
    return 1.0 / seconds if fps == 0.0 else 0.9 * fps + 0.1 / seconds


def start_query_reader(lines: "queue.Queue[str]") -> None:
    """Background thread: every non-empty line typed in the terminal becomes the new query."""
    def run() -> None:
        for line in sys.stdin:
            if line.strip():
                lines.put(line.strip())
    threading.Thread(target=run, daemon=True).start()


def latest(lines: "queue.Queue[str]", current: str) -> str:
    newest = current
    while not lines.empty():
        newest = lines.get_nowait()
    return newest


def open_writer(path: Path, cap: cv2.VideoCapture) -> cv2.VideoWriter:
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise OSError(f"cannot open video writer for {path}")
    return writer


def run(args: argparse.Namespace) -> None:
    s = settings_from_args(args)
    embedder = load_embedder(args.model, args.vision_tokens)
    detector = YOLO(args.detector)
    cap = open_source(args.source)
    writer = open_writer(args.output, cap) if args.output else None
    lines: "queue.Queue[str]" = queue.Queue()
    start_query_reader(lines)

    query_text: str = args.query
    query: np.ndarray = encode_query(embedder, query_text)
    states: dict[int, TrackState] = {}
    fps: float = 0.0
    t_start = time.perf_counter()
    for frame_idx, frame in enumerate(read_frames(cap)):
        t0 = time.perf_counter()
        new_text = latest(lines, query_text)
        if new_text != query_text:
            query_text, query = new_text, encode_query(embedder, new_text)
            logger.info("query changed", extra={"query": query_text})

        res = process_frame(embedder, detector, frame, frame_idx, states, query, query_text, fps, s)
        states = res.states
        fps = next_fps(fps, time.perf_counter() - t0)
        if writer is not None:
            writer.write(res.shown)
        if args.display:
            cv2.imshow(WINDOW, res.shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame_idx % 50 == 0:
            logger.info("frame", extra={"frame": frame_idx, "people": res.people, "encoded": res.encoded,
                                        "target": res.target, "score": res.target_score, "fps": round(fps, 1)})
    elapsed = time.perf_counter() - t_start
    print(f"processed {frame_idx + 1} frames in {elapsed:.1f}s ({(frame_idx + 1) / elapsed:.1f} fps)", flush=True)
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face id or local folder of the EmbeddingGemma model")
    p.add_argument("--source", required=True, help="video file path or webcam index")
    p.add_argument("--query", required=True, help="initial English description of the person to find")
    p.add_argument("--output", type=Path, required=False, default=None, help="also write annotated video here")
    p.add_argument("--display", action=argparse.BooleanOptionalAction, required=False, default=True)
    add_tracking_args(p)
    args = p.parse_args()
    if not args.display and args.output is None:
        p.error("--no-display needs --output, otherwise nothing is shown or saved")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
