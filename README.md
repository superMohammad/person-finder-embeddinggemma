<div align="center">

# Person Finder

**Type one sentence, and it finds the person, in a photo gallery or live in a crowd.**

<img src="docs/media/person-finder.gif" alt="Person Finder: a sentence finds the right photos, then boxes the right shopper live in a mall" width="100%">

<sub>Every frame above is real output. The video is made in code with Remotion (<a href="promo/">promo/</a>), and the full-quality MP4 is <a href="docs/media/person-finder.mp4">here</a>.</sub>

</div>

<br>

| | before | after fine-tuning |
|---|:---:|:---:|
| **First guess is the right person** (Rank-1, 200 unseen people) | 40.1% | **71.4%** |
| **Right person on the first screen** (Rank-10) | 75.4% | **93.2%** |
| **Live video**, detect → track → read → box | | **≈ 26 fps** on one RTX 3090 |

## What it is

One embedding model, [EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2), reads both **words** and **photos** into the same 768-number space. A sentence lands next to the photos of the person it describes.

- **Search.** Your sentence is ranked against 1,000 (or 20,505) person photos, and the best match comes first.
- **Live video.** YOLO26 finds people, ByteTrack follows them, the model reads each crop, and the best match gets a green box. You can change the sentence while the video plays.
- **Fine-tuned for the job.** Contrastive training on 37,010 sentence–photo pairs from [RSTPReid](https://github.com/NjtechCVLab/RSTPReid-Dataset), with batches that never hold the same person twice.
- **Stress-tested in a real crowd.** On the [Mall dataset](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html), with 2,000 frames, up to 53 people each and 62,315 hand-labelled heads.

<p align="center"><img src="docs/diagrams/fig1_overview.png" alt="Figure 1: (a) training with contrastive alignment, (b) text-driven person search in live video" width="100%"></p>

## How it compares

**RSTPReid test split** (2,000 descriptions × 1,000 photos, 200 unseen people)

| Method | Venue | Trained on RSTPReid | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **EmbeddingGemma-2, fine-tuned (this repo)** | | ✓ | **71.1** | **87.4** | **93.2** | **54.7** | **29.5** |
| APTM | ACM MM 2023 | ✓ | 67.5 | 85.7 | 91.5 | 52.6 | – |
| RaSa | IJCAI 2023 | ✓ | 66.9 | 86.5 | 91.4 | 52.3 | – |
| RDE | CVPR 2024 | ✓ | 65.4 | 84.0 | 89.9 | 50.9 | 28.1 |
| IRRA | CVPR 2023 | ✓ | 60.2 | 81.3 | 88.2 | 47.2 | 25.3 |
| CLIP ViT-B/16, fine-tuned | AAAI 2024 | ✓ | 56.2 | 78.3 | 86.6 | 43.3 | – |
| EmbeddingGemma-2, zero-shot | measured here | | 40.0 | 65.5 | 75.3 | 28.8 | 9.7 |
| SigLIP B/16, zero-shot | measured here | | 30.5 | 56.4 | 67.0 | 22.0 | 6.5 |
| CLIP ViT-L/14, zero-shot | measured here | | 11.3 | 26.4 | 37.7 | 8.2 | 2.0 |

**Live video, Mall crowd** (same tracks for every model, 4 descriptions, 573 labelled frames)

| Embedder | Top-1 when the person is in view | AUC | Crops / s |
|---|:---:|:---:|:---:|
| **EmbeddingGemma-2, fine-tuned** | **86.2%** | **0.952** | 34 |
| CLIP ViT-B/16, zero-shot | 84.3% | 0.949 | **345** |
| EmbeddingGemma-2, zero-shot | 85.2% | 0.935 | 34 |
| CLIP ViT-L/14, zero-shot | 82.4% | 0.936 | 143 |
| SigLIP B/16, zero-shot | 74.7% | 0.916 | 345 |

<sub>"measured here": same code and split as this repo, in <a href="TBPR_compare.ipynb">TBPR_compare.ipynb</a>. Published rows are copied from the papers (<a href="https://arxiv.org/abs/2308.09911">RDE, Table 6</a>, <a href="https://arxiv.org/abs/2306.02898">APTM</a>, <a href="https://arxiv.org/abs/2308.10045">TBPS-CLIP</a>). EmbeddingGemma-2 is larger than the ViT-B/16 backbones most of them use. Re-encoding in bf16 moves Rank-1 by about ±0.3 (71.4 in the training notebook).</sub>

On long, detailed descriptions, fine-tuning beats every listed method. On short colour cues in video, CLIP is almost as good and 10× faster. The fine-tuned model's clearest video win is the patterned "red and white striped shirt": 96.8% against 86–88% for CLIP.

**New to the topic?** [**Person_Finder_Explained.pdf**](docs/Person_Finder_Explained.pdf) walks through the whole project from A to Z in plain words. For every step it explains what was done, when, and why, with a glossary at the end.

## Run it

```bash
# 1. environment (Python ≥ 3.13, CUDA GPU)
uv venv .venv --python 3.13 && uv pip install --python .venv -r pyproject.toml
mkdir -p models && .venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo26s.pt')" && mv yolo26s.pt models/

# 2. data: RSTPReid into RSTPReid/ (data_captions.json + imgs/); optional: Mall dataset into mall_dataset/

# 3. experiments, in order. Each notebook writes its numbers and plots to outputs/
.venv/bin/jupyter execute --inplace TBPR_ex.ipynb         # zero-shot baseline, caches embeddings
.venv/bin/jupyter execute --inplace TBPR_finetune.ipynb   # ≈ 18 GB GPU, saves models/embeddinggemma2-rstpreid
.venv/bin/jupyter execute --inplace TBPR_mall.ipynb       # the crowd stress test
.venv/bin/jupyter execute --inplace TBPR_compare.ipynb    # CLIP / SigLIP / published methods, RSTPReid and video

# 4. the website: search on /, live video on /video, method on /how
.venv/bin/python demo/server.py --port 8000 \
    --model models/embeddinggemma2-rstpreid --embeddings outputs/finetuned/image_embeddings.npy \
    --video videos/*.mp4 --video-query "a man in a red and white striped shirt"
```

The video demo also runs on its own, in a desktop window or on a webcam (`--source 0`):

```bash
.venv/bin/python demo/track_person.py --model models/embeddinggemma2-rstpreid \
    --source videos/mall.mp4 --query "a woman in a red coat"
```

<details>
<summary><b>What is where</b></summary>

| Path | What |
|---|---|
| `tbpr.py` | shared data loading and metrics: Rank-k, mAP and mINP on the GPU, plus Mall head matching |
| `TBPR_ex.ipynb` → `TBPR_finetune.ipynb` → `TBPR_mall.ipynb` → `TBPR_compare.ipynb` | the four experiments, in the order they were run |
| `demo/` | standard-library web server, search and video pages, and the YOLO + ByteTrack + EmbeddingGemma tracker |
| `docs/diagrams/` | the method figure, generated as SVG by `build_figures.py` and rendered to PNG by `render.py` |
| `docs/explainer/` | the source of the PDF walkthrough |
| `promo/` | the Remotion project behind the GIF |
| `outputs/` | result tables and plots (embeddings and weights are not committed) |

</details>

## Data and credits

- **RSTPReid** (Zhu et al., ACM MM 2021): photos and descriptions for training and evaluation. Get it from the [official repo](https://github.com/NjtechCVLab/RSTPReid-Dataset).
- **Mall dataset** (Chen et al., BMVC 2012): crowd frames and head points. Build `videos/mall.mp4` with `ffmpeg -framerate 10 -i mall_dataset/frames/seq_%06d.jpg -c:v libx264 -pix_fmt yuv420p videos/mall.mp4`.
- **Demo clips**: OpenCV's `vtest.avi`, and the [Intel IoT DevKit sample videos](https://github.com/intel-iot-devkit/sample-videos) (CC BY 4.0).
- **Models**: [EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2), [Ultralytics YOLO26](https://docs.ultralytics.com/), and ByteTrack.

This is research code on public research datasets. Searching for real people in real footage needs consent, a legal basis, and a human who checks every match.
