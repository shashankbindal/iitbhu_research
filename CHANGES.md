# Project Evolution — from a standard deblurrer to a research contribution

What changed, in order, and why. This traces the path from the initial
engineering approach (a pretrained deblurrer dropped into a pipeline) to the
current research direction (a label-free, recognition-supervised restoration
model). See also [`PROPOSAL.md`](PROPOSAL.md) and [`COMPARISON.md`](COMPARISON.md).

---

## 0. Starting point — RT-Focuser as an off-the-shelf block

The browser demo (the VisAssist Deep Vision Pipeline) used **RT-Focuser**
(Wu et al., 2025) as its Stage-2 deblurrer:

- a **standard, general-purpose motion-deblurring** model, pretrained on GoPro,
- **5.85 M** parameters, **23.7 MB** ONNX,
- used **as-is**, as one block in a pipeline (deblur → detect → caption → speak).

It worked as an engineering integration, but it was a generic model doing a
generic job.

## 1. The problem identified

**Supervisor's critique (Prof. Indradeep Mastan):** arranging pretrained models
(RT-Focuser + YOLO + SmolVLM + MiDaS) into a pipeline is *systems engineering, not
a research contribution*. A publishable paper needs a **novel architecture/method**
that is trained and shown to beat baselines.

**Literature survey finding:** every learning-based deblurrer / text-restoration
method (RT-Focuser, NAFNet; TPGSR, TATT, TextSR, TextDiff; TAIR) needs **paired
clean/degraded training data**. But the real target — photos taken by blind users
(VizWiz) — has **no clean ground truth and never can**. So *no existing method is
trained on the data distribution it must actually serve*; they all train on clean
benchmarks and hope it transfers. That gap is the opening.

## 2. The redirection — approved Option A

> Train a **lightweight** restoration model **directly on real blind-captured
> images (VizWiz)** with **no clean ground truth**, supervised by a **frozen OCR +
> self-consistency**, so that **text becomes machine-readable**.

Success metric changed from pixel fidelity (PSNR) to **downstream OCR accuracy
(CER / word-accuracy)** — the thing that actually matters for a blind user reading
a label.

## 3. Architectural change — RT-Focuser → RestoreNet (R_θ)

| | RT-Focuser (before) | RestoreNet / R_θ (after) |
|---|---|---|
| nature | generic, off-the-shelf | purpose-built for text readability |
| params | 5.85 M | **0.44 M** (~13× smaller) |
| ONNX size | 23.7 MB | **~1.8 MB** |
| design | U-shaped CNN | residual U-Net, **depthwise-separable** convs, **zero-init residual head** (starts as identity) |

## 4. Methodological change — the actual novelty (training)

This is the real change; the architecture above is secondary to it.

| | before | after |
|---|---|---|
| training data | paired clean/blurry (GoPro) | synthetic pairs (Stage 1) **+ real VizWiz, no clean target** (Stage 2) |
| needs clean GT | yes | **no** (Stage 2) |
| loss | pixel / PSNR | **L_conf** (frozen-OCR confidence) + **L_vqa** (weak text from VizWiz answers) + **L_reblur** (anti-hallucination consistency) + **L_content** (structure preservation) + edge |
| optimises for | looking sharp | **being readable** |
| new components | — | **ReblurNet** (learned reblur anchor), **recognizer-coupled losses** (gradients flow through a frozen OCR back into the restorer) |

Two-stage recipe:
1. **Stage 1** — supervised pre-train on synthetic degraded text (clean targets
   exist because the data is synthetic) → strong init.
2. **Stage 2** — **label-free** adaptation on real VizWiz using only the four
   losses above. No clean image is ever used.

## 5. Pipeline change

R_θ is a **drop-in replacement** at Stage 2 of the existing browser pipeline —
identical ONNX interface (NCHW float32 in [0,1]), so it swaps straight into
`models/`. The rest of the pipeline (hazard / caption / depth / speech) is
unchanged. (Option A intentionally does **not** modify YOLO — open question to
confirm with the supervisor if a YOLO change is required.)

## 6. What was built (research codebase, separate `iitbhu_research` repo)

| file | role |
|---|---|
| `degrade.py` | synthetic blind-capture degradation (Stage-1 pairs) |
| `metrics.py` | CER / WER / word-accuracy readability metrics |
| `ocr.py` | frozen OCR wrapper (EasyOCR) for eval |
| `data_vizwiz.py` | VizWiz loader + text-question filter + consensus answers |
| `evaluate.py` / `compare.py` | pluggable restoration eval + the paper's comparison table |
| `model.py` | R_θ RestoreNet |
| `dataset.py` | on-the-fly synthetic text pairs |
| `train_stage1.py` | supervised pre-training |
| `recognizer.py` | differentiable recognizer (TrOCR + TinyCRNN stand-in) |
| `losses_stage2.py` | the four label-free losses |
| `train_stage2.py` | label-free VizWiz adaptation (real `--vizwiz` data path) |
| `export_onnx.py` | R_θ → ONNX for the demo |

## 7. Validation milestones reached

- **Capability proof (overfit):** R_θ takes OCR-garbage back to readable —
  mean CER **0.862 → 0.065**, motion blur visibly removed.
- **Label-free machinery:** Stage-2 loop trains R_θ from degraded image +
  recognizer only (no clean target); reblur consistency loss falls.
- **ONNX drop-in:** exported graph matches PyTorch to **6e-08**, dynamic shapes.
- **GPU training:** now running on the RTX 3050 (via `ankur_env`, CUDA) — the
  full Stage-1 run that turns the capability proof into a generalization result.

## 8. Still pending

- Full **Stage-1 GPU run** (in progress) → generalizing R_θ + held-out CER.
- **Stage-2** on real VizWiz with TrOCR (needs the VizWiz download).
- **`compare.py`** with trained weights → the headline table
  (raw / RT-Focuser / R_θ-S1 / R_θ-S2 / oracle).
- Export the trained R_θ and swap it into the browser demo's `models/`.
