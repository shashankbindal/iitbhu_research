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

> ⚠️ **Superseded for the deployable model.** These numbers are from the *pre-Phase-1*
> (over-aggressive) checkpoint on the older, smaller synthetic set. The deployable,
> identity-preserving checkpoint is evaluated rigorously on the frozen 61-label set in
> **Phase 2 / Phase 5** below — there R_θ-A *ties* RT-Focuser on CER (0.462 vs 0.455)
> and wins on size (13×) and on not destroying sharp photos. Treat the frozen-61
> results as authoritative; this headline is kept for the before/after record.

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

## Phase 0 — metric sanity check (confirmed)

Metric confirmed correct; `word_acc=0` is a **genuine model limitation** at this
training stage, not a metric artifact. Verified by `debug_eval_sanity.py`, which
prints `repr()` of GT vs raw-OCR vs R_θ-OCR for all 8 held-out samples:

- `word_acc` is **exact full-string match** after normalization. GTs are long
  multi-word labels; even R_θ's best output (CER 0.11) still has character errors,
  so exact match is 0 — expected, not a bug. Normalization is applied and not
  masking any true match.
- WER rising (0.786→0.856) while CER falls (0.582→0.455) is **real**: restoration
  makes OCR emit more (and sometimes garbled) word tokens, which lowers character
  error but can raise word-level error. Confirmed by inspecting the strings.
- **CER is the reliable readability metric here**; `word_acc` is too stringent for
  multi-word labels and `WER` is confounded by segmentation. Use CER as primary.

R_θ genuinely improves CER per-sample (e.g. raw `784 8 Ta 277880` → R_θ
`FAFACETAMOL 500-g …` for GT `PARACETAMOL 500mg …`).

## Phase 1 — real-photo smoke test: **R_θ is BROKEN on real photos (STOP gate hit)**

Ran `smoke_test_real.py` on 8 real phone photos (receipts, signs, product boxes).
**Verdict: R_θ output looks broken on real photos** — heavy artifacts, colour
shift, destroyed text. This halts the plan before Phase 2, exactly as the Phase-1
acceptance criterion specifies.

Evidence:
- **Visual:** a sharp, readable bag-shop card → R_θ output is washed to pale-blue
  with dark smearing artifacts and the text destroyed.
- **OCR (all 8 photos):** R_θ consistently *worsens* readability vs raw. E.g. the
  bag sign — raw `Authorised Dealer SAFARI; ARISTOCRAT, SKYBAGS...` → R_θ
  `77777hl Authotfec€ Dealer ARISTOCRAT ; Y,GS...`; the Voltas AC — raw reads a
  full spec list → R_θ collapses to fragments.
- **RT-Focuser, by contrast, preserves the images** (its OCR output ≈ raw, often
  marginally better) — because it is a general deblurrer trained not to destroy
  content.

### Root cause

The Stage-1 training distribution has a fundamental gap: `dataset.py` /
`degrade.py` **always** apply aggressive degradation, so R_θ **never saw an
already-sharp input** and never learned the identity/preservation mapping. Given
a real, mostly-sharp, full-resolution colour photo, it over-processes (tries to
"deblur" sharp content) and produces severe artifacts. It also only ever trained
on 64×256 dark-text-on-light crops, so full-scale colour photos are out of
distribution.

### Required fix before any Config-B architecture work (Phases 3–5)

Revisit the synthetic data pipeline, then retrain Stage 1:
1. Include a substantial fraction (~30–50%) of **near-identity** samples (no / very
   mild degradation, input ≈ target) so the model learns to leave readable content
   alone.
2. Add background/colour and scale diversity (not just dark-on-light text lines).
3. Optionally add a small identity-preservation regulariser.

Building the Config-B FiLM architecture on the current checkpoint would be
investing in a broken foundation — the plan explicitly says to stop here. **Phases
2–5 are paused pending this data-pipeline fix + Stage-1 retrain.**

### Phase 1 RE-TEST after fix: **PASSED** ✅

Applied the fix (`dataset.py`: ~40% near-identity samples [20% exact-sharp, 20%
mild] + ~35% coloured backgrounds + wider scale range), retrained Stage-1 (20k
iters, same recipe), and re-ran `smoke_test_real.py` on the same 8 photos.

**R_θ now preserves real photos** instead of destroying them:
- Bag-shop card: visually identical to raw (was washed-out + smeared before).
- OCR now tracks raw closely on all 8 photos, and is occasionally *cleaner*: e.g.
  the Royal Cuisine receipt — `Bill: RC088507` (R_θ) vs `3ill; RCo88507` (raw),
  timestamp `21:35:46` clean.
- The old (broken) checkpoint is kept as `r_theta_w48_stage1_noident.pth` — this
  before/after is a clean ablation for the paper ("identity-preservation samples
  are necessary; without them the restorer destroys sharp inputs").

Trade-off (honest): on synthetic *heavily-degraded* text, the new model improves
CER less than the old one (0.582→0.529 vs the old 0.582→0.455) — it is less
aggressive because it now also learned to leave sharp content alone. That is the
correct trade: a deployable restorer must not destroy readable images. Demonstrating
strong *improvement* on real BLURRY photos still needs blurry real-photo samples
(the user's set is mostly sharp receipts/signs) — flagged for later.

**Phase 1 gate cleared. Proceeding to Phase 2.**

## Phase 2 — frozen held-out eval set (v2, 61 labels)

Frozen label set: `held_out_eval_v2.json` (61 realistic labels, disjoint from the
procedural training generation; deterministic per-index degradation). This is the
fixed eval set for the Config A vs Config B comparison. Baseline:

| Condition | CER mean | CER std | min | max |
|---|---|---|---|---|
| raw | 0.536 | 0.370 | 0.000 | 1.000 |
| R_θ Config A (Stage-1, identity-preserving) | **0.462** | 0.353 | 0.000 | 1.000 |

Mean CER improves +0.074 (std/√61 ≈ 0.047 standard error → ~1.6σ; modest but real).
The per-sample std (0.35) is the noise floor Config B must clear in Phase 5 to count
as a genuine improvement over Config A.

## Phase 4 — Config B (recognition-guided FiLM) trains correctly ✅

Config B = Config A + a FiLM bottleneck conditioned on a frozen recognizer's view
of the input (`modulation.py`, `RestoreNetConfigB`). Zero-initialised so B ≡ A at
the start; trained with the **identical** recipe, data, seed and iters as A's
retrain (20k, batch 32, width 48) — the only difference is the architecture.

| iter | Config A retrain loss | Config B loss | Config B \|γ\| | Config B \|β\| |
|---|---|---|---|---|
| 1000 | 0.122 | 0.122 | 0.037 | 0.012 |
| 4000 | 0.112 | 0.112 | 0.175 | 0.028 |
| 8000 | ~0.094 | 0.093 | 0.28 | 0.044 |
| 12000 | ~0.085 | 0.085 | 0.353 | 0.056 |
| 16000 | ~0.082 | 0.080 | 0.409 | 0.065 |
| 20000 | 0.082 | **0.078** | **0.41** | 0.067 |

**Acceptance met:** the loss curve tracks Config A's (converges to ~0.08, B
marginally lower), and **γ/β are decisively non-collapsed** — |γ| grew from its
zero-init to ~0.41 (β to ~0.067), i.e. the FiLM module is strongly active, not a
silent no-op. The architecture is correctly wired and the modulation is being used.

## Phase 5 — A vs B head-to-head (frozen 61-label set) — decision: **carry A** 

Identical protocol to Phase 2 (`phase5_compare.py`): same 61 frozen labels, same
EasyOCR, same per-index degradation. Only the architecture differs.

| Condition | CER mean | std | min | max | params / size |
|---|---|---|---|---|---|
| raw | 0.536 | 0.370 | 0.00 | 1.00 | — |
| RT-Focuser (generic deblurrer) | 0.455 | 0.373 | 0.00 | 1.00 | 5.85 M / 23.7 MB |
| **R_θ Config A** | **0.462** | 0.353 | 0.00 | 1.00 | **0.44 M / 1.92 MB** |
| R_θ Config B (FiLM) | 0.498 | 0.381 | 0.00 | 1.00 | 0.51 M / 2.45 MB |
| oracle (sharp) | 0.034 | 0.047 | 0.00 | 0.17 | — |

**Decision.** Config B's CER (0.498) is **higher** than Config A's (0.462): Δ = −0.036,
which is *within* one standard error (0.045) — so B is statistically tied-to-slightly-
worse, and unambiguously **not** the improvement the gate required. Per the plan,
**the architecture change alone did not improve CER at Stage-1 scale, so Config A is
carried forward** to Stage 2. Config B is shelved, not deleted (checkpoint kept).

**Why B didn't help (and what it implies).** The FiLM signal is only as useful as the
recognizer producing it, and at Stage 1 that recognizer is `TinyCRNN` — **frozen but
randomly initialised, with no real reading ability**. So the conditioning is a random
projection of input features, not an informative "what's hard to read here" signal.
The extra capacity let B fit the *training* set marginally better (loss 0.078 < 0.082)
while generalising slightly worse on held-out — textbook mild over-parameterisation
with an uninformative conditioning input. This does **not** refute recognition-guided
modulation; it localises the precondition: **FiLM needs a recognizer that can actually
read.** That is exactly the Stage-2 setup (real frozen TrOCR providing the gradient),
which is where recognition-guided conditioning is the natural thing to re-test — now
as a hypothesis with a concrete, motivated mechanism rather than a random signal.

**Note on the RT-Focuser comparison.** On this rigorous frozen set, the deployable
(identity-preserving) R_θ-A (0.462) and RT-Focuser (0.455) are **statistically tied**
— R_θ's advantage here is **13× fewer params / 12× smaller on disk at equal CER, plus
not destroying sharp real photos** (Phase 1), rather than the CER win reported in the
top headline. That headline (0.263 vs 0.338) is from the *pre-Phase-1* aggressive
checkpoint on the older, smaller synthetic set; **the frozen-61 numbers here supersede
it for the deployable model.** The honest Stage-1 story: R_θ-A matches a 13× larger
generic deblurrer on readability while staying mobile-sized and safe on sharp inputs —
and Stage-2 label-free VizWiz adaptation is where a real readability *win* is expected.

## Phase 6 — Physics-derived unrolled restorer (a third architecture)

Config B's FiLM tweak was still "architecture engineering on top of a black-box
U-Net." The professor's bar requires **mathematical novelty**, not an
architecture variant. Decision: replace the U-Net entirely with a **derived,
unrolled MAP-estimation network** (`unrolled.py`) — every block has a
closed-form mathematical meaning, not just learned layers.

### The model

Observation model: `y = α·(k * x) + β + n`. MAP restoration unrolls
half-quadratic splitting (HQS) into T=4 stages, alternating:

- **Data step** — exact closed-form Wiener deconvolution in the Fourier domain,
  **zero learned parameters**.
- **Prior step** — a tiny shared proximal-denoiser CNN (the only learned part
  besides the kernel estimator).

The unknown blur kernel `k` is predicted per-image, constrained to the
probability simplex (softmax, mixed with a delta via a learned gate) — **provably
a pure blur** (non-negative, sums to 1), never a sharpening or hallucination
operator. This gives two guarantees by construction: identity output for a
delta kernel, and identity-at-init (gate starts at −4 ⇒ mostly delta; denoiser
zero-init). Budget: ~0.08M params (vs 0.44M U-Net, 5.85M RT-Focuser). All claims
backed by runnable self-tests (shape, identity-at-init, Wiener correctness on a
known kernel, simplex validity, gradient flow through every component).

### v1 — no supervision: worse than doing nothing

Trained 20k iterations (survived two silent background-process deaths, most
likely laptop sleep — recovered via a `--resume` flag added to
`train_stage1.py` that reloads the checkpoint's saved iter count).

| Condition | CER mean | std | params / size |
|---|---|---|---|
| raw | 0.536 | 0.370 | — |
| RT-Focuser | 0.455 | 0.373 | 5.85 M |
| R_θ Config A | 0.462 | 0.353 | 0.44 M |
| R_θ Config B | 0.498 | 0.381 | 0.51 M |
| **Unrolled (v1)** | **0.553** | 0.397 | **0.08 M** |
| oracle (sharp) | 0.034 | 0.047 | — |

**Diagnosis (verified via `diag_unrolled.py`, not guessed):** on sharp inputs
the model is correctly near-identity (guarantee holds). On **severely blurred**
inputs, the estimated kernel's centre-tap mass stayed at 0.94–0.96 — almost the
same "barely blurred" kernel it uses on sharp images. **Mechanism:** Wiener
deconvolution is ill-conditioned — dividing by a misestimated kernel's
frequency response amplifies noise. Under a pixel-loss gradient alone, "stay
near identity" is a safer local minimum than committing to strong deconvolution
that risks blowing up on kernel-estimation errors.

### Kernel supervision — partial fix

Since Stage-1 data is synthetic, the **true** blur kernel is known. Added a
direct auxiliary loss supervising the estimator's predicted kernel against the
true kernel (`degrade.py`'s `degrade_with_kernel()`, verified byte-identical to
`degrade()` for existing seeds). Caught two real bugs live via smoke-testing
before committing GPU time: a shape mismatch silently broadcasting into a
nonsensical comparison, and `F.l1_loss`'s default reduction making the
gradient ~625× too weak.

With both fixed, extended smoke tests (300→2500 iters) showed the gate
escaping its saturated init (centre-mass 0.999 → ~0.90–0.92) but then
**plateauing** for 1000+ further iterations — the gate (a scalar "how much to
deviate") learned to open partway, but the kernel **shape** itself (a raw
625-way softmax) settled into a "safe average" response rather than
differentiating per image.

### v2 — NIMA-style quality assessment + encoder warm-start (professor's suggestion)

Professor suggested a NIMA-style (Talebi & Milanfar 2018) quality-assessment
signal to differentiate sharp vs. blurred images. Implemented as an explicit
auxiliary signal feeding the kernel estimator (`nima.py`: distributional
quality head + Earth Mover's Distance loss against a target built from the
known synthetic degradation severity), not a separate demo-side gate.

**Diagnosed the plateau's root cause empirically:** ran the encoder + quality
head **in isolation** (`diag_quality_isolated.py`), decoupled from the
kernel/pixel losses. Trained on the EMD loss alone, it broke out of the same
collapse around iteration ~1800–2000 and reached correlation 0.7–0.9 with true
severity by iteration ~4000–6000 — **proving the encoder architecture CAN
discriminate blur severity.** The plateau was gradient competition (larger-scale
kernel+pixel losses dominating the shared encoder from iteration 1), not an
architectural limit.

**Fix:** warm-start the encoder — pretrain encoder+quality_head on the isolated
EMD loss for 4000 iterations before joint training (`--warmstart_quality`).
Also wired the quality prediction into the gate as an explicit feature and a
zero-init deterministic bias (identity-at-init stays exact).

| Metric | v1 (no supervision) | kernel-sup only (plateaued) | **v2 (kernel+quality sup+warmstart)** |
|---|---|---|---|
| k_centre (severe blur) | 0.94–0.97 (stuck) | 0.90–0.94 (stuck) | **0.37–0.68 (genuinely varying)** |
| loss_k (final) | — | ~1.13–1.16 (flat) | **0.76 (still declining)** |
| loss_q (final) | n/a | stuck ~0.28 | **0.12 (still declining)** |
| **Frozen-61 CER** | 0.553 | *(not run to 20k)* | **0.544** |

v2 trained cleanly to 20,000 iterations with every training-time signal showing
genuine, sustained per-image differentiation. But on the frozen-61 eval, CER
only moved 0.553 → 0.544 (within noise) — still worse than raw and far behind
Config A / RT-Focuser.

**Why the training win didn't transfer.** The kernel/quality supervision
targets are built from synthetic degradation parameters (how much blur/noise/
JPEG/gamma was applied) — a well-posed proxy, but not the real objective (OCR
readability). The model got substantially better at estimating "how degraded
is this image," a different objective from "what pixel change makes text
legible." This also doesn't fix the standing architectural mismatch: JPEG
compression and gamma darkening are nonlinear degradations no single
blur-kernel deconvolution step can represent, regardless of kernel accuracy.

**Separately confirmed deployment blocker:** exported to ONNX
(`export_unrolled_onnx.py`) and tested in a real headless browser against
`onnxruntime-web` (WASM) — the exact runtime the live demo uses. Controlled
comparison: RT-Focuser succeeds in the identical harness; the unrolled model's
`DFT` op (needed for the FFT deconvolution) crashes it. Works in Python
onnxruntime, unsupported in the browser runtime.

**Verdict.** Two full training cycles of the unrolled/physics architecture (v1,
v2) both underperform Config A and RT-Focuser on the frozen-61 set — a
diagnostic-driven negative result, not an under-trained one; v2's training
dynamics were verified healthy at every stage before committing to the full
run. **Recommendation: carry Config A (U-Net) forward as the deployable
backbone** and treat the unrolled architecture's outcome as a documented,
well-diagnosed negative result rather than continuing to iterate without a plan
to close the proxy-objective gap between degradation-severity supervision and
actual readability.

## Artifacts

- `checkpoints/r_theta_w48_stage1.pth` — **Config A, carried forward** (1.79 MB)
- `checkpoints/r_theta_w48.onnx` — self-contained ONNX, 1.92 MB, ready to drop into
  the browser demo's `models/` as a replacement for `rt_focuser.onnx`.
- `checkpoints/r_theta_w48_stage1_configB.pth` — Config B / FiLM (2.45 MB, bundles the
  frozen recognizer). **Shelved** after Phase 5 (no Stage-1 gain); kept for the
  Stage-2 re-test with a real recognizer.
- `checkpoints/r_theta_w48_stage1_noident.pth` — pre-Phase-1 (no identity samples);
  kept as the Phase-1 ablation (restorer destroys sharp inputs without it).
- `checkpoints/r_theta_w48_stage1_unrolled.pth` — Unrolled v2 (kernel+quality
  supervision, warm-started). **Negative result, documented** — see Phase 6.
