"""Convert an exported oscc_calibrated.keras to a self-contained ONNX file and VERIFY it.

Why this exists: tf2onnx.from_function() leaves model constants (e.g. EfficientNet's
Normalization mean/variance) as graph INPUTS, producing an ONNX file that loads but cannot
be run. Here variables are frozen into constants first, then the result is checked against
Keras on random inputs before it is accepted.

Usage:  pip install tf2onnx onnx onnxruntime
        python keras_to_onnx.py /path/to/export/100x      # writes oscc_calibrated.onnx next to the .keras
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
import onnxruntime as ort
import tf2onnx
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def load_float32(path: Path):
    """The notebook trained under mixed_float16, and that policy is stored inside the .keras file, so a plain
    load computes in float16 (which ONNX Runtime on CPU cannot run well). Rebuild the identical graph with the
    float32 policy and load the same weights."""
    with zipfile.ZipFile(path) as z:
        cfg = json.loads(z.read("config.json"))
    cfg = json.loads(json.dumps(cfg).replace("mixed_float16", "float32"))
    m = keras.Model.from_config(cfg["config"])
    m.load_weights(path)
    return m


def convert(folder: Path, atol=1e-3):
    m = load_float32(folder / "oscc_calibrated.keras")
    _, H, W, _ = m.input_shape
    spec = (tf.TensorSpec((None, H, W, 3), tf.uint8, name="image"),)

    @tf.function(input_signature=spec)
    def f(x):
        return m(x)

    frozen = convert_variables_to_constants_v2(f.get_concrete_function())
    out = folder / "oscc_calibrated.onnx"
    tf2onnx.convert.from_graph_def(
        frozen.graph.as_graph_def(),
        input_names=[t.name for t in frozen.inputs], output_names=[t.name for t in frozen.outputs],
        opset=17, output_path=str(out))

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert len(sess.get_inputs()) == 1, f"ONNX still has extra inputs: {[i.name for i in sess.get_inputs()]}"
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        x = rng.integers(0, 255, (2, H, W, 3), dtype=np.uint8)
        ref = m.predict(x, verbose=0)[0]
        got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    assert worst < atol, f"ONNX/Keras mismatch: max |diff| = {worst}"
    print(f"{folder.name}: wrote {out.name} | single input | max |keras-onnx| = {worst:.2e}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(Path(p))
