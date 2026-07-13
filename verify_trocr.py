"""Phase-0 blocker check: does TrOCR load AND route gradients on this GPU box?

The Stage-2 'real recognizer' plan (label-free L_conf/L_vqa AND the Config-B FiLM
re-test) all depend on TrOCR working here. transformers 5.3.0 had a tokenizer bug
on this machine, so verify before building on it. Prints a clear PASS/FAIL.
"""
import sys, time, torch
import torch.nn.functional as F

t0 = time.time()
print("loading microsoft/trocr-small-printed ...", flush=True)
try:
    from recognizer import get_recognizer
    rec = get_recognizer("trocr", device="cuda")
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)
except Exception as e:
    print(f"FAIL: TrOCR did not load: {type(e).__name__}: {e}")
    sys.exit(1)

# A tiny white-on-black 'HELLO'-ish tensor is enough to exercise both losses.
img = torch.rand(2, 3, 48, 192, device="cuda", requires_grad=True)
try:
    lc = rec.confidence_loss(img)
    lc.backward()
    g1 = img.grad.norm().item()
    img2 = torch.rand(2, 3, 48, 192, device="cuda", requires_grad=True)
    lv = rec.text_nll(img2, ["PARACETAMOL 500MG", "DIET COKE"])
    lv.backward()
    g2 = img2.grad.norm().item()
except Exception as e:
    print(f"FAIL: loss/backward errored: {type(e).__name__}: {e}")
    sys.exit(1)

print(f"confidence_loss={lc.item():.3f}  grad-norm={g1:.4f}")
print(f"text_nll       ={lv.item():.3f}  grad-norm={g2:.4f}")
if g1 > 0 and g2 > 0:
    print("PASS: TrOCR loads and routes gradients into the image on GPU.")
else:
    print("FAIL: gradients did not reach the image.")
    sys.exit(1)
