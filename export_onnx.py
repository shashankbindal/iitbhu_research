"""
Stage 3 (integration) — export a trained R_theta to ONNX for the browser demo.

The deployed VisAssist pipeline currently runs the generic RT-Focuser as its
Stage-2 deblurrer. This script exports our trained R_theta to the same ONNX
format so it drops straight in as a stronger, task-specific replacement.

Input/output: NCHW float32 in [0,1], dynamic H and W. Like RT-Focuser, the demo
should pad H,W up to a multiple of 32 before inference and crop back after
(R_theta needs a multiple of 4 for its 2 downsamples; 32 satisfies that and
matches the existing JS padding path).

    python export_onnx.py --ckpt checkpoints/r_theta_w48_stage2.pth --out r_theta.onnx
"""

import argparse
import os
import numpy as np
import torch

from model import RestoreNet


def export(ckpt_path, out_path, opset=17):
    ck = torch.load(ckpt_path, map_location="cpu")
    width = ck.get("width", 48)
    model = RestoreNet(width)
    model.load_state_dict(ck["model"])
    model.eval()

    dummy = torch.rand(1, 3, 64, 256)        # any multiple-of-4 size
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}},
        opset_version=opset,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"exported {out_path}  (width={width}, {size_mb:.2f} MB, opset {opset})")
    return model, width


def verify(model, out_path):
    """Confirm the ONNX graph reproduces the PyTorch output."""
    import onnxruntime as ort
    x = torch.rand(1, 3, 48, 192)
    with torch.no_grad():
        torch_out = model(x).numpy()
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["output"], {"input": x.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"ONNX vs PyTorch max abs diff: {max_diff:.2e}", "OK" if max_diff < 1e-4 else "MISMATCH")
    # also confirm a non-multiple-of-32 but multiple-of-4 size runs
    x2 = np.random.rand(1, 3, 64, 320).astype(np.float32)
    sess.run(["output"], {"input": x2})
    print("dynamic-shape run OK (64x320)")
    return max_diff < 1e-4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/r_theta_w16_stage2.pth")
    p.add_argument("--out", default="checkpoints/r_theta.onnx")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    model, _ = export(args.ckpt, args.out, args.opset)
    verify(model, args.out)


if __name__ == "__main__":
    main()
