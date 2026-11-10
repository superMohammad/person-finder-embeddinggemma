"""Local demo: type a person description, get the top-1 / top-5 RSTPReid images.

Image embeddings are precomputed by the notebooks; the query is encoded at request time
with the same model that produced them.

Run from the project root, then open http://127.0.0.1:8000 (the method page with the figures is /how)

  Fine-tuned model (TBPR_finetune.ipynb):
    .venv/bin/python demo/server.py --port 8000 \\
        --model models/embeddinggemma2-rstpreid --embeddings outputs/finetuned/image_embeddings.npy

  Zero-shot base model (TBPR_ex.ipynb):
    .venv/bin/python demo/server.py --port 8000 \\
        --model google/embeddinggemma-2 --embeddings outputs/image_embeddings.npy

  Add live video grounding on http://127.0.0.1:8000/video (files loop in real time; a number is a webcam):
    ... --video videos/*.mp4 --video-query "a man in a red jacket and dark trousers"
  The video page streams the chosen source with the matching person boxed; the description and the
  source can be changed while it plays. Detector and tracking options are those of demo/track_person.py.
"""
import argparse
import json
import logging
import random
import threading
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from live_video import LiveGrounder, UnknownSource, VideoWorkerFailed, make_sources, thumbnail
from track_person import add_tracking_args, settings_from_args

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_JSON: Path = PROJECT_ROOT / "RSTPReid" / "data_captions.json"
IMG_DIR: Path = PROJECT_ROOT / "RSTPReid" / "imgs"
DEMO_DIR: Path = Path(__file__).resolve().parent
PAGES: dict[str, tuple[Path, str]] = {           # route -> (file, content type)
    "/": (DEMO_DIR / "index.html", "text/html; charset=utf-8"),
    "/video": (DEMO_DIR / "video.html", "text/html; charset=utf-8"),
    "/how": (DEMO_DIR / "how.html", "text/html; charset=utf-8"),
    "/static/style.css": (DEMO_DIR / "style.css", "text/css; charset=utf-8"),
    "/static/theme.js": (DEMO_DIR / "theme.js", "text/javascript; charset=utf-8"),
    "/docs/Person_Finder_Explained.pdf": (PROJECT_ROOT / "docs" / "Person_Finder_Explained.pdf", "application/pdf"),
}
DIAGRAM_DIR: Path = PROJECT_ROOT / "docs" / "diagrams"
MALL_FIGURES: dict[str, Path] = {           # route name -> file, for /figures/<name>
    "mall_detection.png": PROJECT_ROOT / "outputs" / "mall" / "results_detection.png",
    "mall_grounding.png": PROJECT_ROOT / "outputs" / "mall" / "grounding_examples.png",
}
QUERY_PROMPT: str = "SearchQuery"
TOP_K: int = 5
GALLERIES: tuple[str, ...] = ("test", "all")
MJPEG_BOUNDARY: str = "frame"
FRAME_TIMEOUT_S: float = 10.0
THUMB_WIDTH: int = 320
THUMB_AT: float = 0.3
VIDEO_OFF: str = "video is off: start the server with --video <file or webcam index>"

logger = logging.getLogger("tbpr-demo")


@dataclass(frozen=True)
class Record:
    pid: int
    img_path: str
    captions: tuple[str, ...]
    split: str


@dataclass(frozen=True)
class Hit:
    rank: int
    img: str
    pid: int
    score: float
    split: str
    captions: tuple[str, ...]


@dataclass(frozen=True)
class SearchResponse:
    model: str
    query: str
    gallery: str
    gallery_size: int
    hits: list[Hit]


@dataclass(frozen=True)
class Index:
    records: list[Record]
    embeddings: torch.Tensor                 # (N, 768), L2-normalized, on model device
    gallery_rows: dict[str, torch.Tensor]    # gallery name -> row indices into `records`
    image_names: frozenset[str]


class BadRequest(ValueError):
    """Client sent an invalid search request."""


def load_records(path: Path) -> list[Record]:
    with open(path, encoding="utf-8") as f:
        raw: list[dict[str, object]] = json.load(f)
    return [Record(int(d["id"]), str(d["img_path"]), tuple(d["captions"]), str(d["split"])) for d in raw]


def load_index(records: list[Record], emb_path: Path, device: torch.device) -> Index:
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path} not found: run the notebook that caches these image embeddings first")
    emb: np.ndarray = np.load(emb_path)
    if emb.shape[0] != len(records):
        raise ValueError(f"{emb_path} has {emb.shape[0]} rows but the dataset has {len(records)} images")
    test_rows: list[int] = [i for i, r in enumerate(records) if r.split == "test"]
    return Index(
        records=records,
        embeddings=torch.from_numpy(emb).float().to(device),
        gallery_rows={
            "test": torch.tensor(test_rows, device=device),
            "all": torch.arange(len(records), device=device),
        },
        image_names=frozenset(r.img_path for r in records),
    )


def load_model(model_id: str) -> SentenceTransformer:
    cuda: bool = torch.cuda.is_available()
    dtype: torch.dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    return SentenceTransformer(
        model_id,
        device="cuda" if cuda else "cpu",
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"audio_config": None},
    )


def parse_search_body(body: bytes) -> tuple[str, str]:
    try:
        payload: dict[str, object] = json.loads(body)
    except json.JSONDecodeError as e:
        raise BadRequest(f"body is not valid JSON: {e}") from e
    query = payload.get("query")
    gallery = payload.get("gallery")
    if not isinstance(query, str) or not query.strip():
        raise BadRequest("'query' must be a non-empty string")
    if gallery not in GALLERIES:
        raise BadRequest(f"'gallery' must be one of {GALLERIES}, got {gallery!r}")
    return query.strip(), str(gallery)


def parse_text_field(body: bytes, field: str) -> str:
    """The non-empty string `field` of a JSON object body."""
    try:
        payload: dict[str, object] = json.loads(body)
    except json.JSONDecodeError as e:
        raise BadRequest(f"body is not valid JSON: {e}") from e
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BadRequest(f"'{field}' must be a non-empty string")
    return value.strip()


def search(model: SentenceTransformer, model_name: str, index: Index, query: str, gallery: str, k: int) -> SearchResponse:
    q: torch.Tensor = model.encode([query], prompt_name=QUERY_PROMPT, normalize_embeddings=True,
                                   convert_to_tensor=True).float()
    rows: torch.Tensor = index.gallery_rows[gallery]
    scores: torch.Tensor = (q @ index.embeddings[rows].T)[0]
    top = scores.topk(k)
    hits: list[Hit] = []
    for rank, (score, local) in enumerate(zip(top.values.tolist(), top.indices.tolist()), start=1):
        rec = index.records[int(rows[local])]
        hits.append(Hit(rank, rec.img_path, rec.pid, round(float(score), 4), rec.split, rec.captions))
    return SearchResponse(model_name, query, gallery, int(rows.numel()), hits)


def random_test_caption(records: list[Record], rng: random.Random) -> dict[str, str | int]:
    rec = rng.choice([r for r in records if r.split == "test"])
    return {"caption": rng.choice(rec.captions), "pid": rec.pid, "img": rec.img_path}


def make_handler(model: SentenceTransformer, model_name: str, index: Index, rng: random.Random,
                 model_lock: threading.Lock, live: LiveGrounder | None,
                 thumbs: dict[str, bytes]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in PAGES:
                file, content_type = PAGES[self.path]
                self._send(HTTPStatus.OK, content_type, file.read_bytes())
            elif self.path.startswith("/imgs/"):
                name = self.path.removeprefix("/imgs/")
                if name not in index.image_names:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown image {name!r}"})
                    return
                self._send(HTTPStatus.OK, "image/jpeg", (IMG_DIR / name).read_bytes())
            elif self.path.startswith("/figures/"):
                self._send_figure(self.path.removeprefix("/figures/"))
            elif self.path == "/api/example":
                self._send_json(HTTPStatus.OK, random_test_caption(index.records, rng))
            elif self.path == "/api/video/status":
                if live is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": VIDEO_OFF})
                    return
                st = live.status_dict()
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR if st["error"] else HTTPStatus.OK, st)
            elif self.path.startswith("/video/thumb/"):
                source_id = self.path.removeprefix("/video/thumb/")
                if source_id not in thumbs:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"no thumbnail for source {source_id!r}"})
                    return
                self._send(HTTPStatus.OK, "image/jpeg", thumbs[source_id])
            elif self.path == "/video/stream":
                if live is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": VIDEO_OFF})
                    return
                self._stream(live)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path in ("/api/video/query", "/api/video/source"):
                self._video_update(self.path, body)
                return
            if self.path != "/api/search":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})
                return
            try:
                query, gallery = parse_search_body(body)
            except BadRequest as e:
                logger.warning("bad search request", extra={"error": str(e), "body": body[:500]})
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
                return
            with model_lock:
                result = search(model, model_name, index, query, gallery, TOP_K)
            logger.info("search", extra={"gallery": gallery, "top1_pid": result.hits[0].pid})
            self._send_json(HTTPStatus.OK, asdict(result))

        def _send_figure(self, name: str) -> None:
            """GET /figures/<name>: a rendered diagram from docs/diagrams or a Mall result plot, nothing else."""
            diagrams = {p.name: p for p in DIAGRAM_DIR.glob("fig*.png")}
            file = diagrams.get(name) or MALL_FIGURES.get(name)
            if file is None or not file.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown figure {name!r}"})
                return
            self._send(HTTPStatus.OK, "image/png", file.read_bytes())

        def _video_update(self, path: str, body: bytes) -> None:
            """POST /api/video/query {query} or /api/video/source {source}: retarget the live video."""
            if live is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": VIDEO_OFF})
                return
            try:
                if path == "/api/video/query":
                    status = live.set_query(parse_text_field(body, "query"))
                else:
                    status = live.set_source(parse_text_field(body, "source"))
            except (BadRequest, UnknownSource) as e:
                logger.warning("bad video request", extra={"path": path, "error": str(e), "body": body[:500]})
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
                return
            self._send_json(HTTPStatus.OK, asdict(status))

        def _stream(self, grounder: LiveGrounder) -> None:
            """MJPEG: one JPEG part per processed frame until the client disconnects."""
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = 0
            try:
                while True:
                    seq, jpeg = grounder.wait_frame(seq, FRAME_TIMEOUT_S)
                    self.wfile.write(f"--{MJPEG_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                     f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                logger.info("video stream client left", extra={"client": self.client_address[0], "seq": seq})
            except (VideoWorkerFailed, TimeoutError) as e:
                logger.error("video stream ended", extra={"error": str(e), "seq": seq})

        def _send_json(self, status: HTTPStatus, obj: dict[str, object]) -> None:
            self._send(status, "application/json", json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=False, default="127.0.0.1")
    parser.add_argument("--port", type=int, required=False, default=8000)
    parser.add_argument("--model", required=True,
                        help="Hugging Face id or local folder of the model that produced --embeddings")
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="(N_images, 768) .npy of L2-normalized image embeddings in data_captions.json order")
    parser.add_argument("--video", nargs="+", required=False, default=None,
                        help="optional video files (loop) and/or webcam indexes for the /video page")
    parser.add_argument("--video-query", required=False, default=None,
                        help="initial description to track in --video")
    add_tracking_args(parser)
    args = parser.parse_args()
    if args.video is not None and not args.video_query:
        parser.error("--video needs --video-query (the initial description to track)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.video is not None:
        make_sources(args.video)                 # fail on bad paths before loading the model
    records = load_records(DATA_JSON)
    model = load_model(args.model)
    index = load_index(records, args.embeddings, model.device)
    model_lock = threading.Lock()
    live: LiveGrounder | None = None
    thumbs: dict[str, bytes] = {}
    if args.video is not None:
        sources = make_sources(args.video)
        thumbs = {s.id: thumbnail(s, THUMB_WIDTH, THUMB_AT) for s in sources if not s.path.isdigit()}
        model[0].processor.image_processor.max_soft_tokens = args.vision_tokens
        live = LiveGrounder(model, model_lock, sources, args.detector, settings_from_args(args),
                            args.video_query)
        live.start()
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(model, args.model, index, random.Random(0), model_lock, live, thumbs))
    print(f"ready: open http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
