# Next Steps — Config B (Recognition-Guided FiLM Modulation)

> Target audience: Claude Code, operating inside the `iitbhu_research` repo.
> Read this top to bottom. Execute phases in order. Do not skip acceptance
> criteria — each phase gates the next. Stop and report if a criterion fails;
> do not silently continue to the next phase.

## Context (read-only, do not re-derive)

- Stage 1 (Config A) is trained: `checkpoints/r_theta_w48_stage1.pth`, width-48,
  20k iters, AMP, synthetic paired data, pixel/L1 + content/edge loss.
- Held-out result (n=8, synthetic): raw CER 0.582 → R_θ CER 0.455.
- Headline result (separate synthetic eval set): raw 0.429 → RT-Focuser 0.338 →
  **R_θ 0.263** → oracle 0.095. R_θ beats RT-Focuser at 13x fewer params (0.44M
  vs 5.85M) and 12x smaller ONNX (1.92MB vs 23.7MB).
- ONNX export pipeline (`export_onnx.py`) works and produces
  `checkpoints/r_theta_w48.onnx`.
- Stage 2 (label-free VizWiz adaptation) has NOT started.
- Config B (this plan) = add recognition-guided FiLM modulation to the decoder,
  sourced from a frozen lightweight recognizer's intermediate features. This is
  an architecture change, evaluated against Config A under the **identical**
  Stage-1 recipe first, before touching Stage 2 or VizWiz.
- Hardware: RTX 3050, 6GB VRAM, `ankur_env` (CUDA). Server/Colab Pro available
  as fallback if Config B does not fit locally — do not pre-emptively switch;
  attempt local first per Phase 2 acceptance criteria.

---

## Phase 0 — Metric sanity check (do this before anything else)

**Why:** the held-out eval showed `word_acc=0.000` for both raw and R_θ, and
WER got worse (0.786→0.856) while CER improved. This must be confirmed as a
real model behavior, not an eval bug, before any number from this eval is
trusted as a baseline.

- [ ] In `metrics.py`, locate the word-accuracy and WER computation.
- [ ] Add a debug script `debug_eval_sanity.py` that, for each of the 8
      held-out samples, prints `repr()` of: ground truth string, raw OCR
      output string, R_θ OCR output string.
- [ ] Manually inspect for: case mismatch, trailing whitespace/punctuation,
      unicode normalization differences between prediction and ground truth.
- [ ] If a bug is found, fix it in `metrics.py` and re-run the existing
      held-out eval (do not change the model or data — only the metric code).
- [ ] If no bug is found, record in `RESULTS.md` (create if absent) one
      sentence confirming word_acc=0 was checked and is a genuine model
      limitation at this training stage, not a metric artifact.

**Acceptance criteria:** a written note (in `RESULTS.md`) stating either
"metric bug fixed, re-run CER = X" or "metric confirmed correct, word_acc=0 is
real." Do not proceed to Phase 1 without this note existing.

---

## Phase 1 — Real-photo smoke test (no code changes to the model)

**Why:** all current numbers are on synthetic degradation. Before investing in
a new architecture, confirm the Stage-1 checkpoint isn't doing something
synthetic-degradation-specific that won't transfer.

- [ ] Collect or request 5-10 real phone photos containing text (any source;
      blurry/low-light preferred to match the target distribution).
- [ ] Write `smoke_test_real.py`: loads `r_theta_w48_stage1.pth`, runs each
      real photo through (a) raw, (b) RT-Focuser ONNX if available, (c) R_θ,
      then runs EasyOCR on each output.
- [ ] Print side-by-side OCR text for all three conditions per image. No CER
      computation needed (no ground truth) — this is qualitative only.
- [ ] Save outputs to `smoke_test_outputs/` as images for manual review.

**Acceptance criteria:** a short written judgment call in `RESULTS.md` —
"R_θ output looks reasonable / looks broken on real photos" — based on visual
inspection. If broken (e.g., heavy artifacts, color shift, hallucinated
texture), stop and report before Phase 2; this would indicate the synthetic
degradation pipeline in `degrade.py` needs revisiting before any architecture
work is worth doing.

---

## Phase 2 — Expand held-out eval set

**Why:** n=8 is a go/no-go signal, not a number for a paper or a reliable
architecture-comparison baseline.

- [ ] In `data_vizwiz.py` or the synthetic generation path used for held-out
      eval, expand the held-out label set from 8 to at least 50 strings,
      disjoint from the training set.
- [ ] Re-run the existing eval protocol (raw vs R_θ-Stage1, same OCR) on this
      expanded set.
- [ ] Compute and report: mean CER, std CER, min/max CER across samples (not
      just the mean) for both raw and R_θ.
- [ ] Save this as the new fixed `held_out_eval_v2.json` (list of label
      strings) — this exact file is the eval set Config B will be compared
      against. Do not regenerate it per run; freeze it now.

**Acceptance criteria:** `held_out_eval_v2.json` exists with ≥50 entries, and
`RESULTS.md` has an updated table: raw CER, R_θ-Stage1 CER, with mean/std for
both. This becomes the baseline number for the Config A vs Config B
comparison in Phase 5.

---

## Phase 3 — Implement Config B architecture

**New files:**

- [ ] `modulation.py` — defines `FiLMModulator` module.
  - Input: feature map from the frozen recognizer's conv stack (NOT the
    RNN/CTC output — use the spatial feature map before sequence
    flattening).
  - Output: per-channel scale (`gamma`) and shift (`beta`) tensors, produced
    via a small 1x1-conv + global-pool projection, sized to match the target
    decoder stage's channel count.
  - Apply as: `decoder_feat = decoder_feat * (1 + gamma) + beta`.
  - **v1 scope: inject at exactly one point** — the bottleneck or first
    decoder upsampling stage. Do not attempt multi-scale injection in v1;
    that is a v2 extension if v1 shows a CER improvement.

- [ ] In `recognizer.py`, confirm or add a `TinyCRNN` class with a method that
      exposes its intermediate conv feature map (not just final CTC output).
      **Use TinyCRNN for this, not TrOCR** — TrOCR is reserved for final
      evaluation metrics only, not for repeated use inside the training
      graph (memory cost).

**Modified files:**

- [ ] `model.py` — add a `RestoreNetConfigB` class (or a `use_film: bool`
      constructor flag on the existing `RestoreNet`) that wires
      `FiLMModulator` into the decoder at the single injection point above.
      Keep `RestoreNetConfigA` / the existing class fully intact and
      unmodified — Config A must remain runnable as-is for comparison.

- [ ] `train_stage1.py` — add a `--config {a,b}` CLI flag. When `b`, run the
      identical training recipe (same data, same iters, same optimizer, same
      pixel/L1 + content/edge loss) as Config A, with only the model class
      swapped. Do not change any hyperparameter between A and B — this run
      must be a controlled architecture-only comparison.

**Memory constraints (RTX 3050, 6GB) — implement all of these from the start,
not as a reaction to an OOM:**

- [ ] Batch size 1–2 for Config B training.
- [ ] Mixed precision: wrap forward/backward in `torch.cuda.amp.autocast()`
      with `GradScaler` (should already exist from Stage 1; confirm it's
      applied to the new FiLM path too).
- [ ] Extract the frozen recognizer's modulation features under
      `torch.no_grad()` if the loss function for Config B's Stage-1 training
      does not require gradients through the recognizer (it should not, at
      this stage — Config B here is pixel/L1 + content loss, same as Config
      A; the recognizer is only producing the conditioning signal, not part
      of the loss yet). This is the single biggest memory saver available —
      confirm it's applied before assuming you need gradient checkpointing.
  - [ ] If `no_grad` alone is insufficient and batch size 1 with AMP still
        OOMs: add `torch.utils.checkpoint` around the frozen recognizer's
        forward pass as a second measure.
  - [ ] If it still OOMs at batch size 1 with both measures: stop, report
        the exact OOM point and memory figures, and recommend moving Config
        B training to the server/Colab Pro. Do not silently downgrade the
        architecture (e.g., shrinking FiLM module size) to force a local fit
        without flagging it first.

**Acceptance criteria:** `python train_stage1.py --config b` runs to
completion on the RTX 3050 without OOM (or a clear, reported reason why not),
producing `checkpoints/r_theta_w48_stage1_configB.pth`.

---

## Phase 4 — Validate Config B trains correctly (before comparing to A)

- [ ] Confirm the Config B training loss curve decreases comparably to
      Config A's (0.163→0.103 trajectory) — not necessarily identical, but
      not diverging or stuck. Plot or tabulate both curves side by side in
      `RESULTS.md`.
- [ ] Sanity-check the FiLM module is actually being used: log `gamma`/`beta`
      statistics (mean, std) every N iterations early in training. If
      `gamma≈0` and `beta≈0` throughout (i.e., the modulation collapses to a
      no-op), the conditioning signal isn't reaching the decoder — debug
      before proceeding, this would invalidate any comparison.

**Acceptance criteria:** training completes, loss curve is reasonable, and
`gamma`/`beta` logs show non-trivial (non-collapsed) values at convergence.

---

## Phase 5 — Head-to-head comparison: Config A vs Config B

- [ ] Run the identical eval protocol from Phase 2 (`held_out_eval_v2.json`,
      same EasyOCR instance, same CER/WER/word-acc code from `metrics.py`
      post-Phase-0-fix) on both:
  - `r_theta_w48_stage1.pth` (Config A)
  - `r_theta_w48_stage1_configB.pth` (Config B)
- [ ] Produce a comparison table in `RESULTS.md`:

  | Condition | CER (mean) | CER (std) | Params | ONNX size |
  |---|---|---|---|---|
  | raw | | | — | — |
  | RT-Focuser | | | 5.85M | 23.7MB |
  | R_θ Config A | | | 0.44M | 1.92MB |
  | R_θ Config B | | | (report) | (report) |
  | oracle | | | — | — |

**Decision gate:**

- If Config B's mean CER is meaningfully lower than Config A's (not just
  within noise — compare against the std computed in Phase 2): Config B is
  the architecture to carry forward into Stage 2. Update `RESULTS.md` with
  this conclusion explicitly.
- If Config B does not beat Config A, or the difference is within the
  Phase-2 std: do not discard Config B yet, but do not block Stage-2 progress
  on it either. Report the result plainly, and default to carrying Config A
  into Stage 2 while leaving Config B as a documented, ablation-table entry
  ("architecture change alone did not improve CER at Stage-1 scale") — this
  is still a valid, honest result for the paper.

**Acceptance criteria:** the table above is complete and filled in with real
numbers, and a one-paragraph decision is written in `RESULTS.md` stating which
architecture proceeds to Stage 2 and why.

---

## Explicitly out of scope for this plan (do not start without separate sign-off)

- Multi-scale FiLM injection (v2 extension) — only after Phase 5 shows v1
  works.
- Meta-auxiliary test-time adaptation (Config C/D) — separate plan, separate
  risk profile, likely needs server GPU. Do not begin until Config B's
  Phase-5 decision is made.
- Swapping any checkpoint into the live browser demo's `models/` directory —
  not until Stage 2 is complete AND a real-photo (not synthetic) evaluation
  has been done, per the existing caveat already on record.
- Stage 2 (VizWiz label-free adaptation) training itself — this plan stops at
  the Phase 5 decision gate. Stage 2 is the next plan, written after this one
  completes.