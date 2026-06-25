# Running the training on your RTX 3050

All code is built and CPU-verified. The actual training must run on your own
machine — the development sandbox can only fetch CPU torch builds, so it cannot
use the GPU even though the 3050 is physically present. On your machine (normal
internet) these are the exact steps.

## 0. Install the CUDA build of torch (one time)

The default `pip install torch` gives a CPU build. Install the CUDA build:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

Then the rest of the deps:
```bash
pip install -r requirements.txt
```

Every training script auto-detects CUDA and enables AMP — no flags needed.

## 1. Stage 1 — supervised pre-training (synthetic, no download)

```bash
python train_stage1.py --width 48 --iters 60000 --batch 32 --crop 64 256
```

- Fits 6 GB comfortably (model is 0.44M params). If memory is tight: `--batch 16`.
- Prints a raw-vs-R_theta OCR CER check at the end on held-out images.
- Saves `checkpoints/r_theta_w48_stage1.pth`.

## 2. Get the VizWiz text subset

Download VizWiz-VQA images + annotations from https://vizwiz.org/tasks-and-datasets/vqa/
(`train.json` / `val.json` + the image folders). The loader auto-filters to
text-reading questions and extracts consensus answers — no manual labelling.

## 3. Stage 2 — label-free adaptation on real VizWiz

```bash
python train_stage2.py \
    --recognizer trocr \
    --init checkpoints/r_theta_w48_stage1.pth \
    --vizwiz_images /path/to/vizwiz/train --vizwiz_ann /path/to/vizwiz/train.json \
    --width 48 --iters 20000 --batch 16
```

- Uses **no clean ground truth** — the four label-free losses only.
- If the TrOCR tokenizer errors on your transformers version, the code already
  requests `use_fast=False`; if it still fails, pin a known-good version:
  `pip install "transformers==4.44.*"`.
- 6 GB note: TrOCR-small (~62M, frozen) + R_theta + activations. If OOM, drop to
  `--batch 8`, or run the recognizer in fp16 (autocast already does this under AMP).
- Saves `checkpoints/r_theta_w48_stage2.pth`.

## 4. Produce the comparison table (the paper's headline result)

```bash
python compare.py \
    --s1 checkpoints/r_theta_w48_stage1.pth \
    --s2 checkpoints/r_theta_w48_stage2.pth \
    --rtfocuser ../models/rt_focuser.onnx
```

Scores raw / RT-Focuser / R_theta(S1) / R_theta(S2) / oracle on the same eval set
+ OCR. **The key result is S2 < S1 < raw in CER** (label-free adaptation beats
synthetic-only beats no restoration), and S2 vs RT-Focuser shows the task-specific
model beating the generic deblurrer on text.

> Best practice: also evaluate on a held-out **real** VizWiz/TextVQA set, not only
> the synthetic eval, for the paper's main numbers.

## 5. Deploy the trained model into the browser demo

```bash
python export_onnx.py --ckpt checkpoints/r_theta_w48_stage2.pth --out r_theta.onnx
# copy r_theta.onnx into the demo repo's models/ and point Stage 2 of deep-vision.js at it
```

## Expected rough timings on a 3050 (6 GB)

| step | rough time |
|---|---|
| Stage 1 (60k iters, batch 32) | ~1–2 h |
| Stage 2 (20k iters, batch 16, TrOCR) | ~3–5 h (TrOCR forward/backward dominates) |
| compare.py / export | minutes |

Start with shorter runs (`--iters 10000`) to sanity-check the curves before the
full runs.
