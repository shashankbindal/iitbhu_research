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
| Stage 1 | synthetic supervised pre-training of `R_theta` | not started |
| Stage 2 | label-free adaptation on VizWiz (`L_conf`, `L_reblur`, `L_content`, `L_vqa`) | not started |

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

## Next

1. **Stage 1** — `train_stage1.py`: supervised pre-train `R_theta` on synthetic
   `degrade()` pairs (we have GT). Device-agnostic, AMP, text-crop batches for 6GB.
2. **Stage 2** — `train_stage2.py`: label-free adaptation on VizWiz with the four
   losses; the differentiable recognizer (PARSeq/TrOCR) for `L_conf` lives here.
3. Plug `R_theta` into `evaluate.py`; produce the raw / RT-Focuser / ours / oracle table.
