# Progress log: RT-Focuser baseline → physics-derived unrolled restorer

A chronological account of the research direction, what was tried, what worked,
what didn't, and why — from the original demo pipeline through today's
in-progress kernel-supervision fix.

---

## 0. Starting point: the demo pipeline (pre-research)

The original VisAssist demo chained pretrained, off-the-shelf models client-side
in the browser (a "Deep Vision Pipeline"): a generic deblurrer (**RT-Focuser**,
5.85M params, 23.7 MB ONNX) feeding an OCR/VQA stage, to help blind users read
text from a phone photo with no server round-trip.

**Professor's critique (Prof. Indradeep Mastan):** *"Arranging pretrained models
into a pipeline is engineering, not a research contribution."* A publishable
result needs a trained architecture/method shown to beat baselines — not just
composed frozen models.

---

## 1. Research pivot: Option A (approved)

**Direction:** train a lightweight restoration model **R_θ** directly for
downstream **OCR readability (CER)**, not pixel fidelity — eventually **label-free**
on real VizWiz photos (no clean ground truth), supervised by a frozen recognizer's
confidence + weak VQA answers. Two repos split: research (`iitbhu_research`) vs.
demo (`VisAssist_frontend`).

---

## 2. Stage 1, Config A — plain U-Net restorer

`model.py: RestoreNet` — depthwise-separable residual U-Net, zero-init head
(starts as identity), 0.44M params, ONNX 1.92 MB.

- **Phase 0** (metric sanity): confirmed CER is the trustworthy readability
  metric; `word_acc=0` on multi-word labels is real, not a metric bug.
- **Phase 1** (real-photo smoke test) — **STOP gate hit**: R_θ **destroyed** sharp
  real photos (washed-out, smearing artifacts) because training data was *always*
  aggressively degraded — the model never learned to leave already-sharp content
  alone. **Fixed** `dataset.py`: added ~40% near-identity training samples (20%
  exact-sharp, 20% mildly-degraded) + colour/scale diversity. Retrained; re-test
  **PASSED** — R_θ now preserves real photos.
- **Phase 2** (frozen 61-label held-out eval, `held_out_eval_v2.json`, fixed
  degradation seeds): baseline established.

| Condition | CER mean | std | params / size |
|---|---|---|---|
| raw | 0.536 | 0.370 | — |
| RT-Focuser | 0.455 | 0.373 | 5.85 M / 23.7 MB |
| **R_θ Config A** | **0.462** | 0.353 | **0.44 M / 1.92 MB** |
| oracle (sharp) | 0.034 | 0.047 | — |

Config A **statistically ties** RT-Focuser on CER while being **13× smaller** —
and, unlike the pre-fix checkpoint, doesn't destroy sharp inputs.

---

## 3. Config B — Recognition-Guided FiLM (shelved)

**Idea:** condition the U-Net's bottleneck on a frozen recognizer's view of the
input via FiLM (`modulation.py`, zero-init so B ≡ A at start).

- **Phase 3/4:** implemented, trained 20k iters identical recipe to A. Loss
  tracked A's curve; **γ/β decisively non-collapsed** (|γ| grew 0 → 0.41) — the
  architecture trains correctly.
- **Phase 5 (decision gate):** CER **0.498**, *worse* than Config A (0.462),
  within noise of "not better." **Diagnosis:** the Stage-1 recognizer
  (`TinyCRNN`) is frozen but **randomly initialised** — an uninformative
  conditioning signal, so the extra capacity mildly overfits. **Shelved**, not
  deleted: flagged for re-test at Stage 2 with a real recognizer (TrOCR).

---

## 4. Second pivot: physics-derived unrolled restorer

Even Config B's FiLM tweak was still "architecture engineering on top of a
black-box U-Net." Discussed with the user: the professor's bar requires
**mathematical novelty**, not an architecture variant. Decision: replace the
U-Net entirely with a **derived, unrolled MAP-estimation network** — every block
has a closed-form mathematical meaning, not just learned layers.

### The model (`unrolled.py`)

Observation model: `y = α·(k * x) + β + n`. MAP restoration unrolls
half-quadratic splitting (HQS) into T=4 stages, alternating:

- **Data step** — exact closed-form Wiener deconvolution in the Fourier domain,
  **zero learned parameters**.
- **Prior step** — a tiny shared proximal-denoiser CNN (the only learned part
  besides the kernel estimator).

The unknown blur kernel `k` is predicted per-image, constrained to the
probability simplex (softmax, mixed with a delta via a learned gate) — **provably
a pure blur** (non-negative, sums to 1), never a sharpening or hallucination
operator. This gives two guarantees by construction, not by hoping training data
teaches them: identity output for a delta kernel, and identity-at-init (gate
starts at −4 ⇒ mostly delta; denoiser zero-init).

**Budget:** ~0.08M params (vs 0.44M U-Net, 5.85M RT-Focuser).

All claims backed by runnable self-tests in `unrolled.py` (shape, identity-at-init,
Wiener correctness on a known kernel, simplex validity, gradient flow through
every component) — all passing.

### Training — two silent crashes, recovered

Launched the standard 20k-iter Stage-1 recipe (`--config u`). The background
process **died silently twice** (no Python error, process just vanished) at
roughly the 30–45 minute mark each time — most likely the laptop sleeping and
killing all processes, `nohup`/background-task tracking notwithstanding. Added a
`--resume` flag to `train_stage1.py` (reloads checkpoint + saved iter count) and
resumed both times without losing meaningful progress. Training completed
cleanly at 20,000/20,000 iterations, loss converged to **0.0913**.

### Phase U3 — frozen-61 comparison: **worse than doing nothing**

| Condition | CER mean | std | params / size |
|---|---|---|---|
| raw | 0.536 | 0.370 | — |
| RT-Focuser | 0.455 | 0.373 | 5.85 M |
| R_θ Config A | 0.462 | 0.353 | 0.44 M |
| R_θ Config B | 0.498 | 0.381 | 0.51 M |
| **Unrolled (v1)** | **0.553** | 0.397 | **0.08 M** |
| oracle (sharp) | 0.034 | 0.047 | — |

**Diagnosis (verified, not guessed):** wrote `diag_unrolled.py` to inspect
before/after images and the estimated kernel per sample.

- On an already-sharp input, the model is correctly near-identity (guarantee
  holds — `CETIRIZINE 10MG` sample visually unchanged).
- On **severely blurred** inputs, the estimated kernel's centre-tap mass stayed
  at **0.94–0.96 — almost the same "barely blurred" kernel it uses on sharp
  images.** The model converged to a systematically **under-corrective,
  conservative** solution rather than hallucinating artifacts.
- **Mechanism:** Wiener deconvolution is ill-conditioned — dividing by a
  misestimated kernel's frequency response amplifies noise. Under a pixel-loss
  gradient alone, "stay near identity" is a safer local minimum than committing
  to strong deconvolution that risks blowing up on kernel-estimation errors.
  Consistent with its plateaued loss (0.091) being meaningfully higher than the
  U-Net's (0.082) — it isn't fitting the restoration task as well, it's playing
  safe.

---

## 5. Current work: kernel-supervision fix (in progress, not yet resolved)

**Principled fix identified:** Stage-1 data is synthetic, so the **true** blur
kernel used to degrade each sample is known. Add a direct auxiliary loss
supervising the estimator's predicted kernel against the true kernel — bypassing
the ill-conditioned FFT-division gradient path entirely.

**Implemented** (this session):
- `degrade.py` — refactored into a shared core so `degrade_with_kernel()` returns
  the true composed blur kernel (motion ⊛ defocus, or identity if neither was
  applied) alongside the image. **Verified byte-identical** to the original
  `degrade()` for the same seed (critical: the frozen-61 eval set's
  reproducibility must not change).
- `dataset.py` — `SyntheticTextPairs(return_kernel=True)` also yields the true
  kernel tensor per sample.
- `unrolled.py` — `forward(x, return_kernel=True)` returns the kernel
  **un-detached** so a loss can backprop into the estimator directly.
- `train_stage1.py` — new `--w_kernel` weight; `loss = pixel_loss + w_kernel *
  kernel_loss`.

**Two real bugs caught and fixed live via smoke-testing before committing GPU
time to a full run:**
1. Shape mismatch (`kpred` was `(B,1,25,25)`, true kernel was `(B,25,25)`) —
   silently broadcast into a nonsensical `(B,B,25,25)` comparison.
2. `F.l1_loss`'s default reduction averages over **all** elements (batch × 625
   kernel bins), making the loss ~625× smaller than intended and the resulting
   gradient far too weak to move the estimator. Fixed to sum over kernel bins,
   mean over batch (proper 0–2 "distance between distributions" scale).

**Diagnostic smoke tests (300 → 2500 iters), what was actually observed:**

1. First 300 iters: centre-mass climbed 0.983 → 0.999 (wrong direction) while
   `loss_k` sat at ~1.16–1.18 — looked like the −4 gate-bias saturation
   (`σ(1−σ) ≈ 0.018 × 0.982 ≈ 0.0177`) was blocking gradient entirely.
2. Extended to 2500 iters: it **wasn't** permanently stuck — centre-mass
   reversed and dropped 0.999 → ~0.90–0.92 by iter 700 (the gate *did* escape
   saturation; implied gate opening `s` went from ~0.018 to ~0.10, a 5× move).
3. But from iter 700 → 1700 (1000 further iterations), centre-mass and
   `loss_k` both **plateaued** — oscillating in a 0.90–0.94 / 1.13–1.16 band
   with no further improvement. Confirmed genuine (not early noise) by running
   well past the point where it should have kept moving if it were still
   learning.

**Reading of the plateau:** the gate (a single scalar "how much to deviate from
identity") did learn to open partway — that part of the fix worked. What's
*not* moving is the kernel **shape** itself: predicting a raw `25×25=625`-way
softmax per image is a much harder regression than the scalar gate, and a
uniform-ish spread is the "safe average" response across a batch containing
wildly different true kernels (various motion angles/lengths, disks, deltas) —
it reduces loss on average without committing to any one image's actual shape.
That average distance is plausibly exactly the ~1.1–1.2 floor being observed.

**Not yet done:** re-parametrising the kernel head to predict a handful of
physically-meaningful scalars (motion length + angle, defocus radius, mixture
weight) — matching how the training kernels are actually generated — instead
of a raw 625-way softmax. This turns kernel estimation from "learn an arbitrary
625-dim distribution" into "regress 3–4 numbers with known targets," which
should be far better-conditioned and learn faster. **Not implemented or tested
yet** — this is the next candidate step, not a result.

## v2 — NIMA-style quality assessment + encoder warm-start (professor's suggestion)

Professor suggested using a NIMA-style (Talebi & Milanfar 2018) quality-assessment
signal to help differentiate sharp vs. blurred images. Implemented as an
**explicit, well-supervised auxiliary signal into the kernel estimator**, not a
separate demo-side gate (`nima.py`: distributional quality head + Earth Mover's
Distance loss against a target built from the KNOWN synthetic degradation
severity — blur-kernel spread + noise/JPEG/lowlight severity, since Stage-1 data
is synthetic and ground truth is exact).

**Diagnosed root cause of the v1 plateau, empirically (not guessed):** ran the
encoder + quality head **in isolation**, decoupled from the kernel/pixel losses
(`diag_quality_isolated.py`). Same architecture, same data — but trained on the
EMD loss alone, it broke out of the same "predict the average" collapse around
iteration ~1800–2000 and reached correlation 0.7–0.9 with true severity by
iteration ~4000–6000. **This proved the encoder architecture CAN discriminate
blur severity** — the v1/kernel-only plateau was gradient competition (the
larger-scale kernel+pixel losses dominating the shared encoder's gradient from
iteration 1), not an architectural incapacity.

**Fix, directly motivated by that measurement:** warm-start the encoder —
pretrain encoder+quality_head on the isolated EMD loss for 4000 iterations
(cheap, no FFT/prox) before joint training (`--warmstart_quality`,
`train_stage1.py`). Also wired the quality prediction back into the kernel
estimator as an explicit feature AND a deterministic (zero-init, so
identity-at-init is untouched) additive bias into the gate's logit — giving the
gate an immediate signal instead of relying solely on escaping its saturated
−4-bias region via gradient alone.

**Result: training diagnostics improved dramatically; the eval result did not.**

| Metric | v1 (no supervision) | kernel-sup only (plateaued) | **v2 (kernel+quality sup+warmstart)** |
|---|---|---|---|
| k_centre (severe blur) | 0.94–0.97 (stuck) | 0.90–0.94 (stuck) | **0.37–0.68 (genuinely varying)** |
| loss_k (final) | — | ~1.13–1.16 (flat) | **0.76 (still declining)** |
| loss_q (final) | n/a | stuck ~0.28 | **0.12 (still declining)** |
| **Frozen-61 CER** | 0.553 | *(not run to 20k)* | **0.544** |

v2 trained cleanly to 20,000 iterations with every training-time signal showing
genuine, sustained per-image differentiation — a real, verified fix for the
diagnosed gradient-competition problem. But on the frozen 61-label OCR-readability
eval, CER only moved 0.553 → 0.544 (within noise) and remains **worse than doing
nothing** (raw 0.536) and far behind Config A (0.462) / RT-Focuser (0.455).

**Why the training win didn't transfer to readability.** The kernel/quality
supervision targets are built from the *synthetic degradation parameters*
(how much blur/noise/JPEG/gamma was applied) — a well-posed, exactly-known
regression target — not from OCR outcomes. The model got substantially better
at estimating "how degraded is this image mathematically," which is a different
objective from "what pixel change makes text legible to an OCR engine." More
accurate, more aggressive per-image deconvolution trades the old failure mode
(too conservative, safe, low variance) for a new one (real but not
readability-aligned correction) without a net gain. This also does not fix the
standing architectural mismatch flagged earlier: JPEG compression and gamma
darkening are nonlinear degradations that no single blur-kernel deconvolution
step can represent, regardless of how accurately the kernel itself is estimated.

**Status: two full training cycles of the unrolled/physics architecture (v1, v2)
have now been evaluated on the frozen-61 set and both underperform Config A and
RT-Focuser.** This is a real, diagnostic-driven negative result, not an
under-trained one — v2's training dynamics were verified healthy at every stage
before committing to the full run. Continuing to iterate on this architecture
(e.g. the still-unimplemented physically-parametrised kernel head) is possible
but has repeatedly not translated training-metric gains into OCR gains; the
live-demo integration work (ONNX export, browser wiring) is also blocked
separately — the model's `DFT` op is unsupported by `onnxruntime-web`'s WASM
backend (confirmed via a controlled headless-browser test: RT-Focuser succeeds
in the identical harness, the unrolled model fails with a raw WASM exception).

---

## Where things stand

| Model | CER (frozen-61) | Params | Verdict |
|---|---|---|---|
| raw | 0.536 | — | baseline |
| RT-Focuser | 0.455 | 5.85 M | pretrained, not novel |
| **R_θ Config A (U-Net)** | **0.462** | **0.44 M** | ties RT-Focuser, 13× smaller — current best deployable |
| R_θ Config B (FiLM) | 0.498 | 0.51 M | shelved — recognizer was uninformative at Stage 1 |
| Unrolled v1 (physics, no supervision) | 0.553 | 0.08 M | worse than raw — conservative kernel estimation |
| Unrolled v2 (kernel+NIMA quality sup, warm-started) | 0.544 | 0.08 M | training diagnostics fully fixed; **eval result unchanged** |

The physics-derived architecture was the strongest *novelty* candidate on paper
(closed-form math, provable guarantees, smallest footprint) and both of its
diagnosed training-time failures were found and fixed with real evidence (not
guesses) — yet **two full training cycles now both underperform the plain U-Net
and RT-Focuser on actual OCR readability.** The pattern across v1→v2 is
informative: fixing *how well the model estimates its own degradation* (kernel
accuracy, quality-score correlation) did not fix *how much it helps OCR read the
text*. Plausible root cause: the supervision targets (synthetic degradation
severity) are only a proxy for the real objective (readability), and the
architecture's core operation — single global blur-kernel deconvolution — cannot
represent the nonlinear parts of the actual degradation (JPEG blocking, gamma
darkening) no matter how accurate the kernel becomes. Un-implemented follow-ups
(physically-parametrised kernel head) would still need to clear that same gap.
**Recommendation: carry Config A (U-Net) forward as the deployable backbone**
(it already ties RT-Focuser at 13× fewer params) and treat the unrolled
architecture's negative result as a documented, well-diagnosed finding for the
paper rather than continuing to iterate on it without a plan to close the
proxy-objective gap. The live-demo integration is separately blocked either way:
the model's `DFT` op is unsupported by `onnxruntime-web`'s WASM backend
(confirmed via a controlled headless-browser test — RT-Focuser succeeds in the
identical harness, the unrolled model fails with a raw WASM exception).
