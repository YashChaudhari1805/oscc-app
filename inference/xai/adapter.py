"""Model adapters: one interface for ONNX serving and Keras research.

    logit(x)          calibrated pre-sigmoid logit (NOT log(p/(1-p)) of a float32 probability, which saturates at ~16)
    features(x)       (logit, F[N,h,w,C]) when the exact head decomposition has been verified
    grad_logit(x)     (logit, d logit / d x)  Keras backend only
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .onnx_head import HeadNotDecomposable, derive_xai_model, verify_identity


@dataclass
class Head:
    w: np.ndarray        # [C]
    b: float
    scale: float         # logit = scale * (b + mean_ij w.F_ij)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


class ModelAdapter:
    img_hw: tuple
    has_cam = False
    has_grad = False
    head: Head | None = None
    name = "model"

    def logit(self, x):
        raise NotImplementedError

    def prob(self, x):
        return sigmoid(self.logit(x))

    def features(self, x):
        raise NotImplementedError("exact decomposition not available for this adapter")

    def grad_logit(self, x):
        raise NotImplementedError("gradients need the Keras backend")

    def batched(self, fn, x, bs=8):
        return np.concatenate([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


class OnnxAdapter(ModelAdapter):
    def __init__(self, onnx_path, img_hw, require_cam=False):
        import onnxruntime as ort
        self.img_hw, self.name = tuple(img_hw), Path(onnx_path).parent.name
        so = ort.SessionOptions(); so.intra_op_num_threads = 2
        self.cam_error = None
        try:
            blob, w, b, logit_t, feat_t = derive_xai_model(onnx_path)
            self.sess = ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])
            self._names = [o.name for o in self.sess.get_outputs()]
            self._in = self.sess.get_inputs()[0].name
            probe = self.sess.run(None, {self._in: np.zeros((1, *self.img_hw, 3), np.uint8)})
            fshape = probe[-1].shape
            self._nchw = fshape[1] == len(w) and fshape[-1] != len(w)
            s, err = verify_identity(self._run, w, b, self.img_hw)
            self.head, self.has_cam, self.identity_error = Head(w, b, s), True, err
        except Exception as e:                         # no onnx package / unknown head / identity fails: degrade, loudly
            if require_cam:
                raise
            self.cam_error = f"{type(e).__name__}: {e}"
            self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
            self._in = self.sess.get_inputs()[0].name
            self._names = [o.name for o in self.sess.get_outputs()]

    def _run(self, x):
        out = self.sess.run(None, {self._in: x})
        p, logit, F = out[0], out[-2], out[-1]
        if self._nchw:
            F = np.transpose(F, (0, 2, 3, 1))
        return p.reshape(len(x), -1)[:, 0], np.asarray(logit, np.float64).reshape(len(x), -1)[:, 0], F

    def logit(self, x):
        if self.has_cam:
            return self.batched(lambda b: self._run(b)[1], x)
        p = self.batched(lambda b: np.asarray(self.sess.run(None, {self._in: b})[0], np.float64).reshape(len(b), -1)[:, 0], x)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def features(self, x):
        if not self.has_cam:
            raise NotImplementedError(self.cam_error or "no decomposition")
        outs = [self._run(x[i:i + 8]) for i in range(0, len(x), 8)]
        return np.concatenate([o[1] for o in outs]), np.concatenate([o[2] for o in outs])


class KerasAdapter(ModelAdapter):
    """Research backend: exact logits, exact features, true gradients, and weight randomisation for sanity checks."""

    def __init__(self, keras_path, img_hw=None, model=None):
        import keras
        import tensorflow as tf
        self.tf, self.keras = tf, keras
        if model is None:
            with zipfile.ZipFile(keras_path) as z:
                cfg = json.loads(z.read("config.json"))
            cfg = json.loads(json.dumps(cfg).replace("mixed_float16", "float32"))   # same rebuild trick as scripts/keras_to_onnx.py
            model = keras.Model.from_config(cfg["config"])
            model.load_weights(str(keras_path))
        self.model = model
        self.img_hw = tuple(img_hw or model.input_shape[1:3])
        self.name = Path(keras_path).parent.name if keras_path else "keras"
        self._build()

    def _build(self):
        keras, tf = self.keras, self.tf
        m = self.model
        logit_t = m.get_layer("p_oscc").input
        gap = m.get_layer("gap")
        feat_t = gap.input
        # float entry point: the tensor just after the uint8->float cast, so gradients w.r.t. pixels exist
        entry = m.get_layer("rescaling").input
        self.fmodel = keras.Model(entry, [logit_t, feat_t])
        dense = m.get_layer("bin")
        W, B = dense.get_weights()
        scale = 1.0
        for nm in ("scale_bin",):
            try:
                scale = float(m.get_layer(nm).scale)
            except Exception:
                pass
        # p_oscc may read the Dense or the scale layer; logit_t already includes any scale, so fit s numerically
        self.head = Head(W[:, 0].astype(np.float64), float(B[0]), 1.0)
        self._float_in_shape = self.fmodel.input_shape
        self.has_cam = self.has_grad = True
        x = np.random.default_rng(0).integers(0, 255, (2, *self.img_hw, 3)).astype(np.uint8)
        lg, F = self._fwd(x.astype(np.float32))
        pred = self.head.b + (F.astype(np.float64) @ self.head.w).mean((1, 2))
        self.head.scale = float(pred @ lg / (pred @ pred))
        self.identity_error = float(np.abs(self.head.scale * pred - lg).max())
        if self.identity_error > 2e-3 * max(1, np.abs(lg).max()):
            raise HeadNotDecomposable(f"Keras head identity fails: {self.identity_error}")

    def _fwd(self, xf):
        lg, F = self.fmodel(xf, training=False)
        return np.asarray(lg, np.float64).reshape(len(xf), -1)[:, 0], np.asarray(F, np.float32)

    def logit(self, x):
        return self.batched(lambda b: self._fwd(b.astype(np.float32))[0], x)

    def features(self, x):
        o = [self._fwd(x[i:i + 8].astype(np.float32)) for i in range(0, len(x), 8)]
        return np.concatenate([a for a, _ in o]), np.concatenate([f for _, f in o])

    def grad_logit(self, x):
        tf = self.tf
        xt = tf.convert_to_tensor(np.asarray(x, np.float32))
        with tf.GradientTape() as t:
            t.watch(xt)
            lg = self.fmodel(xt, training=False)[0]
            lg = tf.reshape(lg, (-1,))
        g = t.gradient(lg, xt)
        return np.asarray(lg, np.float64), np.asarray(g, np.float32)

    def clone_with_randomised(self, layer_prefixes, seed=0):
        """Copy of this adapter whose conv/dense kernels in the given layer-name prefixes are re-initialised (sanity check)."""
        keras = self.keras
        clone = keras.models.clone_model(self.model)
        clone.set_weights(self.model.get_weights())
        rng = np.random.default_rng(seed)
        for layer in clone.layers:
            if any(layer.name.startswith(p) for p in layer_prefixes) and layer.weights:
                ws = layer.get_weights()
                new = []
                for w in ws:
                    if w.ndim >= 2:
                        fan = np.prod(w.shape[:-1]) + w.shape[-1]
                        lim = np.sqrt(6.0 / fan)
                        new.append(rng.uniform(-lim, lim, w.shape).astype(w.dtype))
                    else:
                        new.append(w)
                layer.set_weights(new)
        return KerasAdapter(None, self.img_hw, model=clone)


def load_adapter(model_dir, backend="auto", img_hw=None):
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "serving_config.json").read_text())
    hw = tuple(img_hw or cfg["img_hw"])
    onnx_p, keras_p = model_dir / "oscc_calibrated.onnx", model_dir / "oscc_calibrated.keras"
    if backend == "keras" or (backend == "auto" and not onnx_p.exists()):
        return KerasAdapter(keras_p, hw)
    return OnnxAdapter(onnx_p, hw)
