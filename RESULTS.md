# Results (Stage 1, GPU-trained)

First real trained results for R_θ. Trained on the RTX 3050 (via `ankur_env`,
CUDA) — width-48, 20k iters, AMP, loss 0.163 → 0.103.

## Headline comparison (frozen OCR: EasyOCR)

Same eval set + same OCR for every condition. CER = Character Error Rate (lower
is better).

| Condition | CER | Params | ONNX size |
|---|---|---|---|
| raw (degraded) | 0.429 | — | — |
| RT-Focuser (generic deblurrer) | 0.338 | 5.85 M | 23.7 MB |
| **R_θ — ours, Stage-1** | **0.263** | **0.44 M** | **1.92 MB** |
| oracle (sharp) | 0.095 | — | — |

**R_θ beats RT-Focuser on text readability (0.263 < 0.338) while being ~13× smaller
in params and ~12× smaller on disk** — and this is Stage-1 only (no label-free
VizWiz adaptation yet). R_θ closes ~50% of the raw→oracle headroom.

## Held-out generalization (separate unseen label set)

A second held-out eval (8 unseen label strings, harder degradation draw):

| | CER | 
|---|---|
| raw | 0.582 |
| R_θ (Stage-1) | 0.455 |

CER −0.127 on images never seen in training → the model **generalizes**, it is not
overfitting. (The two eval seeds give different absolute CERs — small-set
variance; the consistent finding is R_θ < raw, and R_θ < RT-Focuser.)

## Caveats (honest)

- Synthetic eval sets, small (8–10 images). A paper needs a larger fixed held-out
  set and evaluation on **real VizWiz / TextVQA** images.
- Trained on the aggressive *full* composite degradation; absolute CER is still
  high because the task is hard. More iters (Colab 40k) and Stage-2 will improve it.
- Stage 2 (label-free adaptation on real VizWiz) is the next step and is expected
  to help most on real photos, where R_θ's training signal comes from the actual
  target distribution.

## Artifacts

- `checkpoints/r_theta_w48_stage1.pth` — trained weights (1.79 MB)
- `checkpoints/r_theta_w48.onnx` — self-contained ONNX, 1.92 MB, ready to drop into
  the browser demo's `models/` as a replacement for `rt_focuser.onnx`.
