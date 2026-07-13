"""
Isolated diagnostic: can the DegradationEstimator's encoder (3 stride-2 convs +
global average pool) + QualityHead learn to discriminate blur severity AT ALL,
decoupled from the kernel loss / pixel loss / FFT deconvolution?

Motivation: a joint run (train_stage1.py --config u) showed the quality head's
mean predicted score stuck at ~5.0-5.15 (the distribution's midpoint) for 1500
iterations despite loss_q being a well-posed, directly-supervised EMD loss with
no known ill-conditioning (unlike the kernel/gate path). Two possible causes:
  (a) gradient competition in the joint multi-task loss — the shared encoder's
      features get pulled toward whatever the LARGER kernel/pixel losses want.
  (b) the encoder architecture itself (global-average-pooled conv features)
      can't extract a blur-severity signal from this data at all.
This script isolates the encoder+QualityHead with ONLY the EMD loss, run for
many more iterations (cheap — no FFT/prox), to distinguish (a) from (b).
"""
import numpy as np
import torch
import torch.nn as nn

from unrolled import DegradationEstimator
from nima import emd_loss, severity_to_quality_dist, QualityHead
from dataset import SyntheticTextPairs

device = "cuda" if torch.cuda.is_available() else "cpu"
est = DegradationEstimator(ksize=25, ch=32).to(device)
# only the encoder + quality_head need to train for this diagnostic
opt = torch.optim.Adam(list(est.enc.parameters()) + list(est.quality_head.parameters()), lr=2e-4)

ds = SyntheticTextPairs(length=6000 * 32, crop=(48, 192), seed=1,
                        return_kernel=True, ksize=25)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

print("Isolated encoder+QualityHead training (EMD loss only, no kernel/pixel/FFT)")
running = 0.0
for it, (deg, sharp, kgt, qgt) in enumerate(loader, 1):
    deg, qgt = deg.to(device), qgt.to(device)
    h = est.enc(deg).flatten(1)
    pred = est.quality_head(h)
    loss = emd_loss(pred, qgt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    running += loss.item()
    if it % 200 == 0:
        with torch.no_grad():
            pred_score = QualityHead.expected_score(pred)
            true_score = QualityHead.expected_score(qgt)
            # correlation between predicted and true score THIS BATCH -- the
            # real test: not just "loss is low" but "does it track per-image".
            corr = np.corrcoef(pred_score.cpu().numpy(), true_score.cpu().numpy())[0, 1]
        print(f"  iter {it:5d}  loss_q={running/200:.4f}  "
              f"pred_score(mean/std)={pred_score.mean().item():.2f}/{pred_score.std().item():.2f}  "
              f"true_score(mean/std)={true_score.mean().item():.2f}/{true_score.std().item():.2f}  "
              f"corr={corr:.3f}", flush=True)
        running = 0.0
    if it >= 6000:
        break

print("\nDiagnosis: corr near 0 with pred_score.std() collapsing toward 0 => "
      "architecture can't discriminate (cause b). corr rising toward 1 with "
      "pred_score.std() tracking true_score.std() => it CAN learn in isolation, "
      "so the joint run's problem is gradient competition (cause a).")
