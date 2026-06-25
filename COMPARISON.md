# RT-Focuser vs. RestoreNet (R_θ) vs. the deployed pipeline

A side-by-side of (1) the **previous** deblurrer the live demo uses, (2) the
**new** research architecture built in this project, and (3) how each sits in the
**current pipeline**. Written to make the actual contribution legible — and to be
honest about what is proven vs. pending GPU training.

---

## 1. Head-to-head: the architecture change

| Dimension | **RT-Focuser** (previous) | **RestoreNet / R_θ** (new, this project) |
|---|---|---|
| Role | generic motion deblurrer, used as-is | text-**readability** restorer, purpose-built |
| Origin | Wu et al., 2025 — pretrained, off-the-shelf | designed + trained here |
| Architecture | U-shaped CNN | lightweight residual U-Net, depthwise-separable convs, **zero-init residual head** (starts as identity) |
| Parameters | **5.85 M** | **0.44 M** (width 48) — ~13× smaller |
| ONNX size | **23.7 MB** | **~1.8 MB** (full) / 0.16 MB (smoke) |
| Training data | GoPro paired blurry/sharp | synthetic pairs (Stage 1) **+ real VizWiz photos (Stage 2)** |
| Needs clean ground truth? | **Yes** (paired GT required) | **No** for Stage 2 — label-free |
| Training objective | pixel / PSNR (image fidelity) | **readability**: frozen-OCR loss `L_conf` + `L_vqa` + reblur + content (+ edge) |
| Optimizes for | *looking* sharp (30.67 dB PSNR on GoPro) | *being readable* (downstream CER / word-accuracy) |
| Target domain | general motion blur (driving, photography) | blind-captured **text / document** photos |
| Composite low-light + noise + JPEG | motion-blur focused | modelled explicitly in the degradation pipeline |
| Research novelty | **none** — generic model dropped into a pipeline (the critique) | **label-free, recognition-supervised restoration on real blind-captured data** (the contribution) |
| Mobile / 6 GB-trainable | inference-only port | yes — designed for it (0.44 M, AMP, text crops) |
| Status in this project | **deployed** in the browser demo now | **built + CPU-verified**; full training pending the RTX 3050 |

### Why "optimizes for sharp" vs "optimizes for readable" matters

RT-Focuser minimises pixel error against a sharp reference, which makes images
*look* better but is not the same as making text *machine-readable* — and it
cannot be trained at all on the real target data (blind users' photos) because
no sharp reference exists for them. R_θ is supervised by what a frozen OCR can
actually read, so it improves the metric that matters (CER), and it can learn
directly on VizWiz where RT-Focuser structurally cannot.

---

## 2. Where each sits in the pipeline

**Current deployed browser pipeline (Deep Vision Pipeline, in `VisAssist_frontend`):**

```
image → Stage 1 quality check → Stage 2 RT-Focuser (deblur) → Stage 3 hazard/VLM/depth → fuse → speak
                                  ▲ generic, 23.7 MB, PSNR-trained
```

**With the research model (drop-in replacement at Stage 2):**

```
image → Stage 1 quality check → Stage 2 R_θ (readability restore) → Stage 3 … → fuse → speak
                                  ▲ task-specific, ~1.8 MB, readability-trained, trained on real VizWiz
```

The interface is identical (NCHW float32 in [0,1], ONNX), so `export_onnx.py`
produces a file that swaps straight into `models/` — no pipeline rewiring. The
rest of the pipeline (YOLOS hazard, SmolVLM caption, MiDaS depth, speech) is
unchanged; only the deblur block is upgraded.

> Scope note: the research model is **Option A**, which intentionally does **not**
> modify YOLO — it focuses on the restoration stage where the defensible research
> gap is. The deployed pipeline still uses YOLOS for hazards as before.

---

## 3. What is genuinely new (and what isn't)

**New / the contribution**
- Training restoration with **no clean ground truth**, via frozen-recognizer
  gradients + reblur/content self-consistency — enabling learning on real
  blind-captured images (VizWiz) that paired-data methods cannot use.
- A **readability** objective (CER) instead of pixel fidelity (PSNR).
- A purpose-built **tiny** restorer (0.44 M) targeted at text, not general scenes.

**Not new (and not claimed as such)**
- The U-Net restoration backbone idea, depthwise-separable convs, reblur
  consistency, recognition-aware restoration in general — each exists in prior
  work. The novelty is the *combination applied to the un-served setting*: tiny,
  label-free, readability-driven, on real blind-user data.

---

## 4. Honest status

| Claim | Evidence |
|---|---|
| R_θ is far smaller than RT-Focuser | measured: 0.44 M vs 5.85 M params; 1.8 MB vs 23.7 MB ONNX |
| R_θ *can* restore readability | capability proof (overfit): mean OCR CER **0.862 → 0.065**, blur visibly removed |
| Label-free training works without clean GT | Stage-2 loop verified: trains R_θ from degraded image + recognizer only, reblur loss falls |
| ONNX drop-in is valid | exported graph matches PyTorch to **6e-08**, dynamic shapes |
| R_θ **beats** RT-Focuser on text, generalises on real VizWiz | **pending the GPU run** — this is the experiment that turns the capability proof into a paper result (`compare.py` produces the table) |

The last row is the one outstanding piece, and it is compute-bound (full Stage-1
pretrain + Stage-2 VizWiz adaptation on the RTX 3050), not design-bound — see
[`TRAIN_ON_GPU.md`](TRAIN_ON_GPU.md).
