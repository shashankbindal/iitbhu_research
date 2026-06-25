# Research module — Label-Free Recognition-Supervised Restoration

Implementation of [`PROPOSAL.md`](PROPOSAL.md): training a
lightweight restoration network to make text in blind-captured images readable,
**with no clean ground truth**, supervised by a frozen OCR + self-consistency.

## Status

| Stage | Component | Status |
|---|---|---|
| Data/eval (Weeks 1–2) | `degrade.py` synthetic blind-capture degradation | done, tested |
| Data/eval | `metrics.py` CER / WER / word-accuracy | done, tested |
| Data/eval | `ocr.py` frozen-OCR wrapper (EasyOCR) | done, tested |
| Data/eval | `data_vizwiz.py` VizWiz loader + text filter | done, logic tested |
| Data/eval | `evaluate.py` pluggable restoration harness | done, baseline numbers produced |
| Stage 1 | `model.py` R_theta lightweight U-Net (0.44M params @ width 48) | done, tested |
| Stage 1 | `dataset.py` on-the-fly synthetic text-crop pairs | done, tested |
| Stage 1 | `train_stage1.py` supervised pre-train (AMP, device-agnostic) | done, loop verified |
| Stage 1 | `verify_capability.py` overfit capability check | **passed: OCR CER 0.862 -> 0.065** |
| Stage 1 | full GPU pre-training run | pending (needs CUDA machine) |
| Stage 2 | `recognizer.py` differentiable recognizer (TrOCR + TinyCRNN stand-in) | done, grad-flow tested |
| Stage 2 | `losses_stage2.py` label-free objective (`L_conf`,`L_reblur`,`L_content`,`L_vqa`) | done, tested (no clean target) |
| Stage 2 | `train_stage2.py` label-free adaptation loop | done, loop verified |
| Stage 2 | full GPU run on real VizWiz with TrOCR | pending (needs CUDA + VizWiz) |
| Stage 3 | `export_onnx.py` R_theta -> ONNX for the demo | done (diff 6e-08, dynamic shapes) |
| Stage 3 | `compare.py` paper comparison table | done, harness verified |

### Stage-2 machinery (CPU, no clean ground truth used anywhere)

The label-free loop runs and trains `R_theta` from only: the degraded image, the
frozen recognizer, and the weak text label. `L_reblur` already falls (0.055 ->
0.010) as the reblur anchor learns. The recognizer signal (`L_conf`/`L_vqa`) is
flat here only because the CPU smoke test uses a random stand-in; the real
TrOCR provides the informative signal on GPU. This verifies the novel mechanism
is correct before the GPU run.

### Capability check (CPU, `verify_capability.py`)

Overfitting 4 fixed motion-blurred label images (width=32, 800 steps) takes
OCR-unreadable text back to readable — a **proof the architecture can do the
task** before spending GPU time (this is overfit, NOT generalization):

| | degraded OCR | restored OCR |
|---|---|---|
| mean CER over 4 images | 0.862 | **0.065** |

`PARACETAMOL 500mg` → degraded reads `#abauaiana #xmi`, restored reads `PARACETAMOL 50Omg`.

## Current baseline (synthetic eval set, EasyOCR, CPU)

| condition | CER | word-acc |
|---|---|---|
| raw (degraded) | 0.568 | 0.000 |
| oracle (sharp) | 0.079 | 0.500 |

Headroom restoration must recover: **0.49 CER**. A trained `R_theta` should land
between these; the fraction of the gap it closes is the headline result.

## Run

```bash
pip install -r requirements.txt
python degrade.py        # writes _selftest/ sharp + degraded samples
python metrics.py        # metric unit tests
python data_vizwiz.py    # loader logic test (no download needed)
python ocr.py            # OCR smoke test on _selftest images (downloads EasyOCR models 1st run)
python evaluate.py       # full harness: raw vs oracle baseline numbers
```

On Windows the modules set `KMP_DUPLICATE_LIB_OK=TRUE` automatically (PyTorch +
EasyOCR OpenMP conflict). GPU: install the CUDA torch build (see `requirements.txt`).

## How the trained model plugs in

`evaluate.py` scores any `restore_fn: img -> img`. Once `R_theta` is trained,
load it and pass `restore_fn = lambda x: model(x)` alongside `identity` (raw) and
the RT-Focuser baseline — same data, same OCR, directly comparable.

## Run Stage 1

```bash
python train_stage1.py --smoke                       # CPU loop check (~min)
python verify_capability.py                          # overfit capability proof
python train_stage1.py --width 48 --iters 60000 --batch 32   # full run (GPU)
```

## Run Stage 2 / 3

```bash
python train_stage2.py --smoke                       # CPU machinery check
# GPU: label-free adaptation with the real recognizer
python train_stage2.py --recognizer trocr --init checkpoints/r_theta_w48_stage1.pth \
    --width 48 --iters 20000 --batch 16
python export_onnx.py --ckpt checkpoints/r_theta_w48_stage2.pth --out r_theta.onnx
python compare.py --s1 ...stage1.pth --s2 ...stage2.pth --rtfocuser ../models/rt_focuser.onnx
```

## Next (all on the RTX 3050)

1. **Full Stage-1 GPU run** — `train_stage1.py --width 48`, tens of thousands of
   iters, for a *generalizing* pre-trained `R_theta`.
2. **Download VizWiz** text subset; point `train_stage2.py`'s data stream at the
   real loader (replacing `synthetic_stream`), run label-free adaptation with TrOCR.
3. **`compare.py`** with trained S1/S2 + RT-Focuser -> the paper's headline table.
4. **`export_onnx.py`** -> drop `r_theta.onnx` into the browser demo (`models/`).
