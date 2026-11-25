"""Expose the pre-sigmoid logit and the last feature map of the exported ONNX model, WITHOUT TensorFlow.

The model ends  features F[h,w,C] -> GlobalAveragePool -> (Dropout: identity at inference) -> Dense(C->1) -> [x scale] -> sigmoid.
Because every step after F is linear, the logit decomposes EXACTLY over spatial positions:

        logit = scale * ( b + (1/hw) * sum_ij  w . F_ij )            (class-activation-map identity)

We add two graph outputs (the Sigmoid's input and the GAP's input), read w and b from the graph's constants, and then
VERIFY the identity numerically on random inputs. If it does not hold to tolerance (a different head, a hidden
non-linearity, a scale we cannot explain) the decomposition is NOT enabled: fail closed, never show an unverified "exact" map.
"""
from __future__ import annotations

import numpy as np


class HeadNotDecomposable(RuntimeError):
    pass


def _const(g, name, prod, inits):
    """Resolve a tensor to a numpy constant, looking through Cast/Identity/Reshape/Squeeze of an initializer."""
    from onnx import numpy_helper
    seen = 0
    while name not in inits and name in prod and seen < 6:
        n = prod[name]
        if n.op_type == "Constant":
            return numpy_helper.to_array(n.attribute[0].t)
        if n.op_type not in ("Cast", "Identity", "Reshape", "Squeeze", "Unsqueeze"):
            return None
        name, seen = n.input[0], seen + 1
    return numpy_helper.to_array(inits[name]).astype(np.float64) if name in inits else None


def derive_xai_model(onnx_path):
    """Returns (serialized_model_bytes, w[C], b, layout) where layout is 'NCHW' or 'NHWC' for the feature output."""
    import onnx
    m = onnx.load(str(onnx_path))
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    inits = {i.name: i for i in g.initializer}
    sig = prod.get(g.output[0].name)
    if sig is None or sig.op_type != "Sigmoid":
        raise HeadNotDecomposable("first output is not a sigmoid")
    logit_t = sig.input[0]

    # walk back to the MatMul/Gemm (Dense) fed by the global-average-pool
    n, bias, scale_nodes = prod[logit_t], None, []
    for _ in range(8):
        if n.op_type in ("MatMul", "Gemm"):
            break
        if n.op_type == "Add":
            consts = [_const(g, i, prod, inits) for i in n.input]
            for i, c in zip(n.input, consts):
                if c is not None and c.size == 1 and bias is None:
                    bias = float(c.reshape(-1)[0])
            nxt = [i for i, c in zip(n.input, consts) if c is None]
            n = prod[nxt[0]] if nxt else None
        elif n.op_type in ("Cast", "Identity"):
            n = prod.get(n.input[0])
        else:
            raise HeadNotDecomposable(f"unexpected op {n.op_type} between Dense and sigmoid")
        if n is None:
            raise HeadNotDecomposable("could not find Dense layer")
    if n.op_type not in ("MatMul", "Gemm"):
        raise HeadNotDecomposable("Dense layer not found")
    W = None
    for i in n.input:
        c = _const(g, i, prod, inits)
        if c is not None and c.ndim == 2:
            W = c
    if W is None or W.shape[1] != 1:
        raise HeadNotDecomposable("Dense weights not found or not a single-logit head")
    if bias is None:
        bias = 0.0
    x = [i for i in n.input if _const(g, i, prod, inits) is None][0]
    k = prod[x]
    for _ in range(6):                                  # look through Cast / Squeeze / Reshape to the pool
        if k.op_type in ("GlobalAveragePool", "ReduceMean"):
            break
        k = prod[k.input[0]]
    if k.op_type not in ("GlobalAveragePool", "ReduceMean"):
        raise HeadNotDecomposable("global average pool not found")
    feat_t = k.input[0]

    for name in (logit_t, feat_t):
        g.output.append(onnx.helper.make_empty_tensor_value_info(name))
    return m.SerializeToString(), W[:, 0].astype(np.float64), float(bias), logit_t, feat_t


def verify_identity(run, w, b, h_w, n_check=4, tol=2e-3, seed=0):
    """run(batch_uint8) -> (p, logit, F_nhwc). Checks logit == s*(b + mean_ij F_ij.w) with ONE constant s; returns s."""
    rng = np.random.default_rng(seed)
    H, W = h_w
    xs = [rng.integers(0, 256, (1, H, W, 3), dtype=np.uint8) for _ in range(n_check)]
    xs += [np.clip(rng.normal(190, 40, (1, H, W, 3)), 0, 255).astype(np.uint8)]
    pred, tgt = [], []
    for x in xs:
        _, logit, F = run(x)
        pred.append(b + (F[0].astype(np.float64) @ w).mean()); tgt.append(float(logit[0]))
    pred, tgt = np.array(pred), np.array(tgt)
    s = float(pred @ tgt / max(pred @ pred, 1e-12))
    err = float(np.abs(s * pred - tgt).max())
    if err > tol * max(1.0, np.abs(tgt).max()):
        raise HeadNotDecomposable(f"logit != scale*(b + mean_ij w.F_ij): max error {err:.3g}")
    return s, err
