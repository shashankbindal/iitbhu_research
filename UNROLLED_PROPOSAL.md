# Deep-Unrolled Blind Restoration for Assistive Vision
### Replacing the encoder–decoder with a network derived from the imaging physics

**Presenter:** Shashank &nbsp;·&nbsp; **Supervisor:** Prof. Indradeep Mastan &nbsp;·&nbsp; IIT BHU

---

## 1. The demand (why this problem, why these constraints)

Blind and low-vision users photograph objects to ask "what is this? what does the
label say?" Their photos are systematically corrupted **because they cannot see the
viewfinder**: no framing check → motion blur from hand-shake; no focus check →
defocus; indoor capture → low light and sensor noise. The VizWiz dataset (Gurari et
al., CVPR 2018) — 31k photos taken by real blind users — documents this: a large
fraction of images have quality issues severe enough to make questions unanswerable,
and ~21% of all questions ask about *reading text* (labels, expiry dates, prices).

The deployment constraint is equally hard: the solution must run **on the user's
phone with no API** (privacy — these photos contain medicine labels, bank letters;
cost; offline reliability). So the restorer must be **well under 1M parameters** and
use only ONNX/mobile-runtime-friendly ops.

Our earlier system arranged pretrained models into a pipeline (detector + deblurrer
+ OCR). The supervisor's critique stands: that is engineering. And our own controlled
experiments confirmed the deeper point — a standard **encoder–decoder U-Net**
(0.44M params, "Config A") trained on synthetic pairs only *ties* the generic
RT-Focuser deblurrer on readability (CER 0.462 vs 0.455 on our frozen 61-label set),
and adding a learned conditioning module (FiLM, "Config B") **did not help**
(CER 0.498). Black-box architecture tweaks are not the axis of improvement.
The novelty must come from **the mathematics of the problem itself**.

## 2. The mathematics

### 2.1 Observation model

A blind-captured photo is not arbitrary; it obeys a physical degradation model:

$$y \;=\; \alpha\,\big(k \circledast x\big) \;+\; \beta \;+\; n \tag{1}$$

where $x$ is the latent clean scene, $k$ an **unknown** blur kernel (camera-shake /
defocus point-spread function), $\alpha,\beta$ a first-order exposure/low-light
correction, and $n$ sensor noise. A physical blur kernel satisfies, by definition:

$$k \ \ge\ 0, \qquad \textstyle\sum_i k_i \;=\; 1 \tag{2}$$

(energy is redistributed, never created — a blur is a *weighted average*).

### 2.2 Restoration as MAP estimation

Given the illumination-corrected observation $y' = (y-\beta)/\alpha$, the maximum-
a-posteriori estimate of $x$ is

$$\hat{x} \;=\; \arg\min_x\; \tfrac{1}{2}\,\lVert k \circledast x - y'\rVert_2^2 \;+\; \lambda\,\Phi(x) \tag{3}$$

— a data-fidelity term from (1) plus an image prior $\Phi$. **Half-quadratic
splitting** decouples the two by introducing an auxiliary $z$:

$$\min_{x,z}\; \tfrac{1}{2}\lVert k \circledast x - y'\rVert^2 \;+\; \tfrac{\mu}{2}\lVert x - z\rVert^2 \;+\; \lambda\,\Phi(z) \tag{4}$$

and alternating two sub-problems, **each with a clear identity**:

**(i) Data step — quadratic in $x$, solved *exactly* in closed form via the FFT**
(convolution diagonalizes in the Fourier basis):

$$x_{t+1} \;=\; \mathcal{F}^{-1}\!\left[\frac{\overline{\mathcal{F}(k)}\cdot\mathcal{F}(y') \;+\; \mu_t\,\mathcal{F}(z_t)}{\lvert\mathcal{F}(k)\rvert^2 \;+\; \mu_t}\right] \tag{5}$$

This is a Wiener-type deconvolution step. It has **zero learned parameters** and is
exact — no network has to *learn* to invert a convolution, the mathematics does it.

**(ii) Prior step — the proximal operator of $\Phi$:**

$$z_{t+1} \;=\; \mathrm{prox}_{\lambda\Phi/\mu_t}(x_{t+1}) \tag{6}$$

For classical priors (TV, wavelet sparsity) this is a denoiser. We therefore
**learn** it as a tiny shared CNN — the *only* learned image operator in the model.

### 2.3 The network = the algorithm, unrolled

We unroll $T{=}4$ iterations of (5)–(6) into a network. Per-stage $\mu_t>0$ are
learned (softplus). The unknown kernel is predicted by a small estimator whose
output is constrained onto the probability simplex:

$$k \;=\; (1-s)\,\delta \;+\; s\cdot\mathrm{softmax}(\ell),\qquad s=\sigma(g) \tag{7}$$

Both $\delta$ (identity kernel) and any softmax output satisfy (2), and the simplex
is convex — so **every representable $k$ is provably a physical blur**. The same
encoder predicts $(\alpha,\beta)$, initialized at $(1,0)$.

### 2.4 Two properties by construction (not by training data)

**Sharp-input identity.** If the estimator outputs $s\to 0$ (i.e. $k=\delta$,
$\alpha{=}1,\beta{=}0$), then $\mathcal{F}(k)\equiv 1$ and (5) reduces to
$x_{t+1} = (\mathcal{F}(y') + \mu\mathcal{F}(z_t))/(1+\mu)$, which with $z_0=y'$
gives $x=y'$ exactly: **a sharp photo passes through unchanged, as a theorem.**
Our U-Net had to *learn* this from 40% identity-augmented data after it
catastrophically destroyed real sharp photos (our Phase-1 failure); here it is
structural.

**Identity at initialization.** The gate starts at $s\approx0.02$ and the prox-CNN's
last conv is zero-initialized, so the untrained network is $\approx$ identity
(verified: max deviation 0.009). Training can only improve on "do no harm".

### 2.5 Label-free training on real data (Stage 2, unchanged)

VizWiz has no clean ground truth. The same model trains label-free because the
**estimated $k$ is simultaneously the data-fidelity operator and the supervision**:

$$\mathcal{L} \;=\; \underbrace{\lVert \hat{k} \circledast \hat{x} - y'\rVert_1}_{\text{self-consistency (re-blur)}} \;+\; \underbrace{\mathcal{L}_{\mathrm{conf}}(\mathrm{OCR}(\hat{x}))}_{\text{frozen-recognizer confidence}} \;+\; \underbrace{\mathcal{L}_{\mathrm{text}}(\mathrm{OCR}(\hat{x}), a)}_{\text{weak VizWiz answers } a} \tag{8}$$

Because $\hat k$ is constrained to be a blur (2), the self-consistency term cannot
be satisfied by a degenerate $\hat{x}$ (an unconstrained reblur CNN could learn to
*sharpen*, making the check vacuous — the constraint is what gives the term teeth).

## 3. Novelty claim (one sentence)

> **A blind, deep-unrolled MAP restorer whose simplex-constrained kernel estimate
> serves simultaneously as the closed-form data-fidelity operator and as the
> label-free self-supervision signal — trained directly on real blind-captured
> photos with no clean ground truth, at 0.08M parameters for on-phone use.**

Relation to prior work (honest): deep unfolding is established (USRNet, Zhang et
al. CVPR 2020 — *non-blind*, needs the true kernel; DWDN etc.), and task-driven
restoration exists. The unclaimed combination is **blind** (kernel estimated, not
given) + **label-free on real assistive data** (no clean GT anywhere) + **the
kernel's dual role** in (5) and (8) + **mobile scale**. The negative results we
already hold (encoder–decoder ties a generic deblurrer; FiLM conditioning fails)
motivate the shift from black-box capacity to structure.

## 4. Model card

| | U-Net (Config A) | RT-Focuser | **Unrolled (ours)** |
|---|---|---|---|
| Derivation | black-box | black-box | MAP objective (3), unrolled HQS |
| Params | 0.44 M | 5.85 M | **0.079 M** |
| Sharp-input safety | learned (data aug) | learned | **guaranteed (§2.4)** |
| Interpretable internals | no | no | $k$, $\alpha$, $\beta$, $\mu_t$ all physical |
| Blur kernel validity | — | — | **provable: simplex (2)** |

## 5. Experimental plan & acceptance gates

1. **Controlled Stage-1 comparison** (running): identical data/recipe/iters as the
   U-Net. Gate: unrolled ≤ U-Net CER on the frozen 61-label set (it has 5.6× fewer
   params; matching already demonstrates the structural prior's value — beating it
   is the target).
2. **Real-photo safety**: the 8-photo smoke test — must preserve sharp photos
   (expected by construction, §2.4).
3. **Stage-2 label-free adaptation on VizWiz** with (8); evaluate OCR/CER on a
   held-out real slice vs raw / RT-Focuser / U-Net.
4. **Ablations** (each isolates one mathematical claim):
   T = 1/2/4 stages · simplex constraint (7) on vs unconstrained kernel ·
   with/without illumination correction · learned $\mu_t$ vs fixed.
5. **Interpretability figure for the paper**: visualize the estimated $\hat k$ per
   image — motion streaks for shaken photos, near-delta for sharp ones. An
   encoder–decoder has nothing comparable to show.

## 6. Risks (declared up front)

- FFT ops in ONNX export are opset-17 (DFT); if the mobile runtime lacks it, the
  data step (5) can be computed by conjugate-gradient with spatial convolutions
  (same maths, no FFT). Deployment path exists either way.
- Uniform-blur assumption: (1) assumes one kernel per image; real shake is mildly
  spatially varying. Declared as scope; patch-wise kernels are the natural v2.
- If Stage-1 CER is *worse* than the U-Net, the honest report is that structure at
  0.08M underfits the synthetic composite degradation — then we scale prox width /
  stages (still ≪ 0.44M) before any conclusion.
