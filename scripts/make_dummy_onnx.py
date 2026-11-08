"""Create tiny stand-in ONNX models (no TensorFlow needed) with the same input/output contract as the real
exports: uint8 image [N,H,W,3] -> p_oscc [N,1]. The 'score' is just a function of mean brightness, so it is
MEANINGLESS: it exists only to test the app plumbing before you copy the real models.
Usage: python scripts/make_dummy_onnx.py models      (pip install onnx onnxruntime)"""
import json, sys
from pathlib import Path
import numpy as np, onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

out = Path(sys.argv[1] if len(sys.argv) > 1 else "models")
H, W = 96, 128
for mag in ("100x", "400x"):
    nodes = [
        h.make_node("Cast", ["image"], ["f"], to=T.FLOAT),
        h.make_node("ReduceMean", ["f"], ["m"], axes=[1, 2, 3], keepdims=0),          # mean brightness per image, shape [N]
        h.make_node("Sub", ["m", "c"], ["s"]),
        h.make_node("Div", ["s", "d"], ["z"]),
        h.make_node("Sigmoid", ["z"], ["p"]),
        h.make_node("Reshape", ["p", "shape"], ["p_oscc"]),                            # -> [N,1]
    ]
    graph = h.make_graph(
        nodes, f"dummy_{mag}",
        [h.make_tensor_value_info("image", T.UINT8, ["N", H, W, 3])],
        [h.make_tensor_value_info("p_oscc", T.FLOAT, ["N", 1])],
        [nh.from_array(np.float32(180.0), "c"), nh.from_array(np.float32(25.0), "d"),
         nh.from_array(np.array([-1, 1], dtype=np.int64), "shape")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)]); model.ir_version = 8
    onnx.checker.check_model(model)
    d = out / mag; d.mkdir(parents=True, exist_ok=True)
    onnx.save(model, d / "oscc_calibrated.onnx")
    json.dump({"magnification": mag, "img_hw": [H, W], "T_bin": 1.0, "grade_enabled": False, "outputs": ["p_oscc"]},
              open(d / "serving_config.json", "w"))
    print("wrote", d)
print("DUMMY models: scores are meaningless (brighter image = higher score). Replace with the real exports.")
