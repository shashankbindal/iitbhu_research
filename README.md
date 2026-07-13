# Label-Free, Recognition-Supervised Image Restoration for Assistive Vision

Research codebase for training a lightweight image-restoration network, **R_θ**,
that improves what blind users' phone photos let an OCR/VQA system read —
optimising for **readability (Character Error Rate)**, not pixel fidelity
(PSNR/SSIM). Companion to the client-side demo at
[VisAssist_frontend](https://github.com/shashankbindal/VisAssist_frontend).

## Why

Existing assistive-vision pipelines lean on generic, pretrained deblurring
models. This project asks whether a small model, **trained specifically for
this problem**, can match or beat a generic pretrained one — and, longer term,
whether it can be trained with **no clean ground truth at all** (Stage 2: real
blind-captured photos have no "sharp" version to supervise against).

## Result summary

| Model | CER, frozen 61-label eval (↓ better) | Parameters | Status |
|---|---|---|---|
| raw (unrestored) | 0.536 | — | baseline |
| RT-Focuser (pretrained, generic) | 0.455 | 5.85 M | baseline |
| **R_θ Config A (U-Net)** | **0.462** | **0.44 M** | **recommended / deployable** |
| R_θ Config B (recognition-guided FiLM) | 0.498 | 0.51 M | shelved — negative result, diagnosed |
| R_θ Unrolled (physics-derived MAP restorer) | 0.544 | 0.08 M | shelved — negative result, diagnosed |
| oracle (sharp ground truth) | 0.034 | — | upper bound |

**R_θ Config A statistically ties the 5.85M-parameter pretrained baseline on
readability at 13× fewer parameters**, and — unlike an early checkpoint trained
without identity-preservation — never destroys already-sharp input. Full
experimental record, including two independently diagnosed negative results,
is in [`RESULTS.md`](RESULTS.md).

## Repository structure

```
degrade.py            synthetic blind-capture degradation (motion blur, defocus,
                       low light, sensor noise, JPEG) with reproducible seeding
dataset.py             on-the-fly synthetic (degraded, sharp) training pairs
data_vizwiz.py         VizWiz-VQA loader + text-question filter (Stage 2 data)
metrics.py              CER / WER / word-accuracy
ocr.py                  frozen-OCR wrapper (EasyOCR)
evaluate.py             pluggable restoration-quality evaluation harness
held_out_eval.py        frozen 61-label benchmark (held_out_eval_v2.json)
                         used for every architecture comparison in this project

model.py                R_theta Config A: lightweight residual U-Net (0.44M params)
                         + Config B: + recognition-guided FiLM modulation
modulation.py           FiLM module used by Config B
unrolled.py             R_theta Unrolled: physics-derived MAP-estimation restorer
                         (closed-form Wiener deconvolution + learned prior, HQS-unrolled)
nima.py                 NIMA-style quality-assessment auxiliary signal (feeds
                         the unrolled model's kernel estimator)
recognizer.py            frozen differentiable recognizer backends (TrOCR, TinyCRNN)
                         — the Stage-2 supervision signal

train_stage1.py         supervised pre-training on synthetic pairs (configs a/b/u)
train_stage2.py         label-free adaptation on real images, no clean target
losses_stage2.py        the label-free objective (L_conf, L_vqa, L_reblur, L_content)

compare.py               generates the RT-Focuser vs R_theta comparison table
phase5_compare.py        Config A vs B vs Unrolled head-to-head on the frozen set
diag_unrolled.py         per-sample visual/OCR diagnostic for the unrolled model
diag_quality_isolated.py isolates the NIMA quality head to test it independent
                         of the kernel/pixel losses
debug_eval_sanity.py     metric sanity check (confirms CER over word-accuracy)
smoke_test_real.py       runs a trained checkpoint on real (non-synthetic) photos
verify_trocr.py          confirms the TrOCR recognizer loads and trains on GPU

export_onnx.py            R_theta (U-Net) -> ONNX, for the browser demo
export_unrolled_onnx.py   R_theta (Unrolled) -> ONNX

RESULTS.md                full experimental record: every phase, every result,
                           including negative results and their root causes
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

On the GPU machine, install the CUDA build of torch separately (see the note
at the top of `requirements.txt`) — the default pip wheel is CPU-only. On
Windows, set `KMP_DUPLICATE_LIB_OK=TRUE` (PyTorch + EasyOCR OpenMP conflict);
the training/eval scripts do not set this for you.

## Reproducing the results

```bash
# Stage-1 training (Config A, the recommended/deployable model)
python train_stage1.py --config a --width 48 --iters 20000 --batch 32 --workers 2

# Other architectures (both are documented negative results — see RESULTS.md)
python train_stage1.py --config b --width 48 --iters 20000 --batch 32   # FiLM
python train_stage1.py --config u --width 48 --iters 20000 --batch 32 \
    --w_kernel 1.0 --w_quality 1.0 --warmstart_quality 4000              # unrolled

# Evaluate any checkpoint on the frozen 61-label benchmark
python held_out_eval.py --ckpt checkpoints/r_theta_w48_stage1.pth

# Full head-to-head comparison table (raw / RT-Focuser / A / B / Unrolled / oracle)
python phase5_compare.py

# Export the recommended model for the browser demo
python export_onnx.py --ckpt checkpoints/r_theta_w48_stage1.pth --out checkpoints/r_theta.onnx
```

`train_stage1.py --smoke` runs a tiny CPU-only version of the training loop end
to end in under a minute, for verifying the pipeline without a GPU.

## Stage 2 — label-free adaptation (scoped, not yet executed)

The project's core novelty claim: fine-tune R_θ on **real VizWiz photos with no
clean ground truth**, supervised only by a frozen recognizer's confidence, weak
crowd-sourced text answers, and a reblur self-consistency term. All
infrastructure is built and unit-tested (`train_stage2.py`, `losses_stage2.py`,
`data_vizwiz.py`) but the full GPU run against real VizWiz data + a real
recognizer has not yet been executed — the recommended next step.

```bash
python train_stage2.py --smoke                        # CPU machinery check, no clean target used
python train_stage2.py --recognizer trocr \
    --init checkpoints/r_theta_w48_stage1.pth \
    --width 48 --iters 20000 --batch 16                # full run, needs VizWiz + GPU
```

## Further reading

[`RESULTS.md`](RESULTS.md) is the complete experimental log: the Phase 0
metric-validation check, the Phase 1 real-photo failure and fix, the frozen
benchmark construction, the Config A result, and the two fully-diagnosed
negative results (Config B / FiLM, and the physics-derived unrolled restorer)
— including the specific, verified root cause behind each.
