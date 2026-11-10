"""Shared data loading and retrieval metrics for RSTPReid text-based person retrieval."""
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Record:
    """One RSTPReid image with its person id, descriptions and split."""

    pid: int
    img_path: str
    captions: tuple[str, ...]
    split: str


@dataclass(frozen=True)
class CaptionTable:
    """All descriptions flattened, aligned with the records they describe."""

    texts: list[str]
    pids: np.ndarray       # (Q,) person id of each description
    img_idx: np.ndarray    # (Q,) index of the described image in `records`
    split: np.ndarray      # (Q,) split name of each description


@dataclass(frozen=True)
class Retrieval:
    """Per-query retrieval outcome against one gallery."""

    first_hit: np.ndarray   # (Q,) 1-based rank of the first correct image
    ap: np.ndarray          # (Q,) average precision
    inp: np.ndarray         # (Q,) inverse negative penalty
    top5: np.ndarray        # (Q, 5) gallery indices of the 5 best matches


def load_records(path: Path) -> list[Record]:
    with open(path, encoding="utf-8") as f:
        raw: list[dict[str, object]] = json.load(f)
    return [Record(int(d["id"]), str(d["img_path"]), tuple(d["captions"]), str(d["split"])) for d in raw]


def flatten_captions(records: Sequence[Record]) -> CaptionTable:
    rows: list[tuple[str, int, int, str]] = [
        (cap, r.pid, i, r.split) for i, r in enumerate(records) for cap in r.captions
    ]
    return CaptionTable(
        texts=[t for t, _, _, _ in rows],
        pids=np.array([p for _, p, _, _ in rows], dtype=np.int64),
        img_idx=np.array([i for _, _, i, _ in rows], dtype=np.int64),
        split=np.array([s for _, _, _, s in rows]),
    )


def truncate(emb: torch.Tensor, dim: int) -> torch.Tensor:
    """Keep the first `dim` Matryoshka dimensions and re-normalize."""
    return torch.nn.functional.normalize(emb[:, :dim].float(), dim=-1)


def retrieve(q_emb: torch.Tensor, q_pids: np.ndarray, g_emb: torch.Tensor, g_pids: np.ndarray,
             dim: int, chunk: int) -> Retrieval:
    """Rank the whole gallery for every query (cosine similarity, GPU, chunked) and score it."""
    q, g = truncate(q_emb, dim), truncate(g_emb, dim)
    qp = torch.from_numpy(q_pids).to(q.device)
    gp = torch.from_numpy(g_pids).to(g.device)
    ranks = torch.arange(1, g.shape[0] + 1, device=g.device, dtype=torch.float32)
    first, ap, inp, top5 = [], [], [], []
    for s in range(0, q.shape[0], chunk):
        order = (q[s:s + chunk] @ g.T).argsort(dim=1, descending=True)
        match = gp[order] == qp[s:s + chunk, None]                    # (B, G) bool
        n_pos = match.sum(dim=1).float()
        if bool((n_pos == 0).any()):
            raise ValueError("a query has no correct image in the gallery")
        hits = match.float()
        first.append(hits.argmax(dim=1) + 1)
        ap.append(((hits.cumsum(dim=1) / ranks) * hits).sum(dim=1) / n_pos)
        last = g.shape[0] - hits.flip(dims=[1]).argmax(dim=1)          # rank of last correct image
        inp.append(n_pos / last.float())
        top5.append(order[:, :5])
    cat = lambda xs: torch.cat(xs).cpu().numpy()
    return Retrieval(cat(first), cat(ap), cat(inp), cat(top5))


def summarize(r: Retrieval) -> dict[str, float]:
    """Rank-1/5/10, mAP and mINP in percent."""
    return {
        "Rank-1": 100 * float((r.first_hit <= 1).mean()),
        "Rank-5": 100 * float((r.first_hit <= 5).mean()),
        "Rank-10": 100 * float((r.first_hit <= 10).mean()),
        "mAP": 100 * float(r.ap.mean()),
        "mINP": 100 * float(r.inp.mean()),
    }


def cmc(r: Retrieval, max_rank: int) -> np.ndarray:
    """Cumulative match curve: % of queries with a correct image within rank 1..max_rank."""
    return np.array([100 * float((r.first_hit <= k).mean()) for k in range(1, max_rank + 1)])


@dataclass(frozen=True)
class MallFrames:
    """The Mall crowd dataset: 2,000 frames of one shopping-mall camera with a point on every head."""

    frame_paths: list[Path]
    heads: list[np.ndarray]   # per frame, (K, 2) head points (x, y) in pixels
    roi: np.ndarray           # (H, W) bool, the annotated floor region


@dataclass(frozen=True)
class HeadMatch:
    """One-to-one matching of person boxes to ground-truth head points in one frame."""

    tp: int
    fp: int
    fn: int
    gt_hit: np.ndarray        # (K,) bool, whether each head point got a box


def load_mall(root: Path) -> MallFrames:
    """Read `mall_gt.mat` (head points) and `perspective_roi.mat` (region mask) from the unzipped dataset."""
    gt = scipy.io.loadmat(root / "mall_gt.mat")["frame"][0]
    roi = scipy.io.loadmat(root / "perspective_roi.mat")["roi"][0, 0]["mask"].astype(bool)
    paths = [root / "frames" / f"seq_{i + 1:06d}.jpg" for i in range(len(gt))]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} Mall frames are missing, first: {missing[0]}")
    return MallFrames(paths, [np.asarray(f["loc"][0, 0], dtype=np.float32) for f in gt], roi)


def head_points(boxes: np.ndarray, head_frac: float) -> np.ndarray:
    """Expected head point of each (x1, y1, x2, y2) box: horizontal centre, `head_frac` of the height below the top."""
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 1] + head_frac * (boxes[:, 3] - boxes[:, 1])], axis=1)


def boxes_in_roi(boxes: np.ndarray, roi: np.ndarray, head_frac: float) -> np.ndarray:
    """(N,) bool: the box's expected head point lies inside the annotated region."""
    pts = head_points(boxes, head_frac).round().astype(np.int64)
    x = pts[:, 0].clip(0, roi.shape[1] - 1)
    y = pts[:, 1].clip(0, roi.shape[0] - 1)
    return roi[y, x]


def match_heads(boxes: np.ndarray, heads: np.ndarray, top_frac: float, head_frac: float) -> HeadMatch:
    """Match boxes to head points one-to-one (Hungarian, closest expected head point first).

    A head may only be matched to a box that contains it within the top `top_frac` of the box height.
    """
    if len(boxes) == 0 or len(heads) == 0:
        return HeadMatch(0, len(boxes), len(heads), np.zeros(len(heads), dtype=bool))
    x1, y1, x2, y2 = (boxes[:, i][:, None] for i in range(4))
    hx, hy = heads[None, :, 0], heads[None, :, 1]
    inside = (hx >= x1) & (hx <= x2) & (hy >= y1) & (hy <= y1 + top_frac * (y2 - y1))
    dist = np.linalg.norm(head_points(boxes, head_frac)[:, None, :] - heads[None, :, :], axis=2) / (x2 - x1)
    rows, cols = linear_sum_assignment(np.where(inside, dist, 1e6))
    ok = inside[rows, cols]
    gt_hit = np.zeros(len(heads), dtype=bool)
    gt_hit[cols[ok]] = True
    tp = int(ok.sum())
    return HeadMatch(tp, len(boxes) - tp, len(heads) - tp, gt_hit)


# Plot style shared by the notebooks: light surface, recessive axes, fixed categorical order
SERIES: list[str] = ["#2a78d6", "#eb6834", "#1baf7a"]   # slot 1 blue, 2 orange, 3 aqua
INK: str = "#0b0b0b"
INK_2: str = "#52514e"
GRID: str = "#e4e3df"
SURFACE: str = "#fcfcfb"
PLOT_RC: dict[str, str | int | float | bool] = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "legend.frameon": False,
    "lines.linewidth": 2, "font.size": 10,
}
