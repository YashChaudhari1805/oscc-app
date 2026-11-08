# Models

Put the exported folders from the training notebook here, so the layout is:

```
models/100x/oscc_calibrated.keras   models/100x/oscc_calibrated.onnx   models/100x/serving_config.json
models/400x/oscc_calibrated.keras   models/400x/oscc_calibrated.onnx   models/400x/serving_config.json
```

(Copy `MyDrive/Oral_Q1/_outputs/export/100x` and `/400x` from Google Drive.)

**ONNX files exported by the first notebook versions are broken** (weights left as graph inputs;
onnxruntime cannot run them). Regenerate and verify them from the `.keras` files:

```
pip install tensorflow-cpu tf2onnx onnx onnxruntime
python scripts/keras_to_onnx.py models/100x models/400x
```

The script fails loudly unless ONNX output matches Keras. The inference service prefers ONNX when
present (small, fast, no TensorFlow needed) and falls back to the `.keras` file (`BACKEND=keras`).

No model files are shipped with this project. `python scripts/make_dummy_models.py models` creates
RANDOM stand-ins for plumbing tests only; their predictions are meaningless.
