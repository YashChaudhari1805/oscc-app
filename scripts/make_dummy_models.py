"""Create randomly-initialised stand-in models with the SAME input/output contract as the
exports from the training notebook (uint8 NHWC in -> p_oscc, p_grade out). For plumbing tests
and demos ONLY: predictions are meaningless. Usage: python make_dummy_models.py ../models [H W]"""
import json, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from keras import layers

out = Path(sys.argv[1]); H, W = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (96, 128)
for mag in ("100x", "400x"):
    inp = layers.Input((H, W, 3), dtype="uint8", name="image")
    x = layers.Rescaling(1.0, name="to_float")(inp)
    base = keras.applications.EfficientNetB3(include_top=False, weights=None, input_tensor=x)
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    b = layers.Dense(1, name="bin")(x); g = layers.Dense(3, name="grade")(x)
    pb = layers.Activation("sigmoid", name="p_oscc")(layers.Rescaling(1.0, name="scale_bin")(b))
    pg = layers.Activation("softmax", name="p_grade")(layers.Rescaling(1.0, name="scale_grade")(g))
    m = keras.Model(inp, [pb, pg], name=f"oscc_calibrated_{mag}")
    d = out / mag; d.mkdir(parents=True, exist_ok=True)
    m.save(d / "oscc_calibrated.keras")
    json.dump(dict(magnification=mag, img_hw=[H, W], T_bin=1.0, grade_enabled=False, outputs=["p_oscc", "p_grade"]),
              open(d / "serving_config.json", "w"))
    try:
        from keras_to_onnx import convert
        convert(d)
    except Exception as e:
        print(mag, "keras written; onnx FAILED:", repr(e)[:200])
