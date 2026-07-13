"""
Export the unrolled physics-derived restorer to ONNX for the browser demo.

Key risk: the model's data step (unrolled.py) uses torch.fft.rfft2/irfft2 with
complex tensors. ONNX added a DFT op in opset 17, but onnxruntime-web (the
browser WASM/WebGPU runtime used by the demo) may not implement it even if the
export itself succeeds in plain onnxruntime (Python/C++). This script exports
and verifies with plain onnxruntime first — a necessary but not sufficient
check for browser compatibility, which still needs a real browser test.

    python export_unrolled_onnx.py --ckpt checkpoints/r_theta_w48_stage1_unrolled.pth --out unrolled.onnx
"""
import argparse
import os
import numpy as np
import torch

from unrolled import UnrolledRestorer


def export(ckpt_path, out_path, opset=18):
    ck = torch.load(ckpt_path, map_location="cpu")
    model = UnrolledRestorer()
    model.load_state_dict(ck["model"])
    model.eval()

    dummy = torch.rand(1, 3, 64, 256)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}},
        opset_version=opset,
    )

    import onnx
    m = onnx.load(out_path)
    ops = sorted({n.op_type for n in m.graph.node})
    print("ops used in exported graph:", ops)
    onnx.save_model(m, out_path, save_as_external_data=False)
    sidecar = out_path + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"exported {out_path}  ({size_mb:.3f} MB, opset {opset})")
    return model, ops


def verify(model, out_path):
    import onnxruntime as ort
    x = torch.rand(1, 3, 48, 192)
    with torch.no_grad():
        torch_out = model(x).numpy()
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["output"], {"input": x.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"ONNX vs PyTorch max abs diff: {max_diff:.2e}", "OK" if max_diff < 1e-3 else "MISMATCH")
    x2 = np.random.rand(1, 3, 64, 320).astype(np.float32)
    sess.run(["output"], {"input": x2})
    print("dynamic-shape run OK (64x320)")
    return max_diff < 1e-3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/r_theta_w48_stage1_unrolled.pth")
    p.add_argument("--out", default="checkpoints/unrolled.onnx")
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()
    model, ops = export(args.ckpt, args.out, args.opset)
    verify(model, args.out)


if __name__ == "__main__":
    main()
