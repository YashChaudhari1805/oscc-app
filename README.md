# OSCC Assistant

A small web app for trying the trained OSCC model: sign in with an e-mail one-time code, upload a
histopathology field image (JPEG/PNG) at 100x or 400x, and get a calibrated model score with quality
warnings and a printable report.

> **Research prototype. Not a medical device. Not validated for clinical use.** The model was trained and
> checked on a single public dataset (patient-level CV, binary Normal vs OSCC, no external validation).
> The UI says so on every screen and on the report.

## Design

Light by default (dark mode via the toggle; the choice is remembered and the OS setting is deliberately ignored).
Hanken Grotesk (light, tight tracking) with Instrument Serif italic for the accent, both bundled locally. The
ribbed arc is a microscope graticule and doubles as the score dial. No external fonts, scripts or images.

## Explainability ("Why this score?")

After a result, **Explain this result** draws an *occlusion-sensitivity* map over the image: small patches are hidden one at a
time (painted with the image's own blank-glass colour), the model re-scores the image, and each patch is coloured by how much
the OSCC log-odds moved relative to a typical patch. Rose = hiding it lowers the OSCC score; teal = hiding it raises it.

* **Why not Grad-CAM:** in this project's own masking test, Grad-CAM-ranked regions were no more important than random ones.
  Occlusion is a direct measurement of the real model's behaviour, and it works on the ONNX file (no TensorFlow at serving time).
* **Self-check on every map:** the top-ranked patches are hidden together and compared with the same number of random
  patches. If the "important" patches do not move the score clearly more than random ones, the UI shows **Map not confirmed**,
  mutes the overlay and badges the image. Failing the check is informative, not an error.
* **Cost / limits:** about 30 model passes (about 4 s here; depends on CPU). `/api/explain` has its own tighter limits
  (`LIMIT_EXPLAIN_PER_MIN=3`, `_PER_HOUR=20`, `_PER_DAY=60`, `_PER_MIN_IP=6`) and one concurrent explanation (`MAX_CONCURRENT_EXPLAIN`).
  Tuning (inference service): `EXPLAIN_ROWS/COLS` (9x12), `EXPLAIN_WINDOW` (3), `EXPLAIN_STRIDE` (2), `EXPLAIN_RANDOM` (4).
* **What it is not:** it shows what the score is sensitive to, not what is cancer. Patches overlap and the model can use
  context, so it can miss spread-out evidence. It is included (with the self-check result) as its own section of the printable report.
* **Tested:** `inference/tests/test_explain.py` checks the algorithm on models with known behaviour (a model that only uses
  one quadrant must highlight that quadrant and pass the check; a model using the whole image equally must not pass; a constant model
  must give a flat map), plus the API on the real models. `gateway/explain_test.go` covers auth, CSRF, separate limits, busy handling and rejection of malformed model output.
  **Not yet validated on real histology images:** please run it on a set of real fields and record how often the self-check passes.

## Architecture

```
Browser (React, Material 3)  ──HTTPS──▶  Caddy (TLS)  ──▶  Gateway (Go, stdlib only)  ──▶  Inference (Python/FastAPI + ONNX Runtime)
                                                            auth · CSRF · rate limits         internal network only,
                                                            upload sanitising · audit         shared-secret header
```

* `inference/`  runs the **same preprocessing as training** (resize + Macenko; verified bit-exact against the
  notebook by a test) and the exported model. It never faces the internet.
* `gateway/`    owns every security decision. Zero third-party Go dependencies.
* `web/`        React + hand-written Material 3 CSS. No UI library, no external fonts/scripts, so the strict CSP holds.

Scope note: the model takes **single microscope fields (about 1600×1200 JPEG)**, not whole-slide images or DICOM,
so this app does too. The WSI/DICOM/OpenSlide parts of the original blueprint do not apply to this model.

## Quick start (local, any OS: Windows, macOS, Linux)

Needs Python 3.10+, Node 18+, Go 1.22+. Run from the project folder (do **not** double-click `.sh` files: on Windows they open in an editor).

```bash
pip install -r inference/requirements.txt onnx          # onnx is only needed for the dummy models
python scripts/make_dummy_onnx.py models                # tiny stand-in models (no TensorFlow). Scores are MEANINGLESS
cd web && npm install && npm run build && cd ..
python scripts/run_dev.py                               # starts inference + gateway; Ctrl+C stops both
```

Open http://127.0.0.1:8080 and sign in as `anything@example.org`. The 6-digit code is printed **in the same terminal**
(look for `DEV_ECHO_OTP sign-in code`). `scripts/run_dev.sh` does the same on macOS/Linux.
If it says Go/Python/port problems, `run_dev.py` prints what to fix.

## With your real models

**Download from Drive** (`MyDrive/Oral_Q1/_outputs/export/`), for each of `100x` and `400x`:

| File | Needed? |
|---|---|
| `oscc_calibrated.keras` | yes, only to build the new ONNX (step 2); it can be deleted afterwards |
| `serving_config.json` | **yes** |
| `oscc_calibrated.onnx` | **no: the one in Drive is broken**, you will create a working one |
| `saved_model/` folder | no (only for TensorFlow Serving) |

1. Put `oscc_calibrated.keras` and `serving_config.json` into `models/100x/` and `models/400x/` (delete the dummy ones first).
2. Create working ONNX files. Easiest is Google Colab, because TensorFlow is painful on Windows: upload
   `scripts/keras_to_onnx.py` to Colab, mount Drive, then run
   `!pip -q install tf2onnx onnx onnxruntime` and
   `!python keras_to_onnx.py /content/drive/MyDrive/Oral_Q1/_outputs/export/100x /content/drive/MyDrive/Oral_Q1/_outputs/export/400x`.
   It rebuilds the model in float32 (the files were trained under mixed precision), converts, and refuses to finish unless
   ONNX matches Keras (prints `max |keras-onnx| = ...e-07`). Then download the two new `oscc_calibrated.onnx` files.
   (On a machine with TensorFlow you can run the same script locally on `models/100x models/400x`.)
3. Final layout: `models/100x/{oscc_calibrated.onnx, serving_config.json}` and the same for `400x`. The `.keras` files are optional.
4. Run `python scripts/run_dev.py`. `ALLOWED_DOMAINS` / `ALLOWED_EMAILS` control who may sign in
   (e.g. PowerShell: `$env:ALLOWED_EMAILS="you@gmail.com"; python scripts/run_dev.py`).
   If a code never arrives, the terminal now says when the address is not on the allow-list.
5. Check that sample images give sensible scores before showing anyone.

## Production

`cp .env.example .env`, fill it in, then `docker compose --env-file .env -f deploy/docker-compose.yml up -d --build`.
The gateway **refuses to start** in production unless: HTTPS cookies are on, SMTP is configured, an allow-list
is set, secrets are ≥ 32 chars, and no dev switches (`ALLOW_ANY_EMAIL`, `DEV_ECHO_OTP`) are enabled.
The compose/Docker files were written but **not executed** where this was authored (no Docker available): test them.

## Security controls

| Area | What it does |
|---|---|
| Verification | E-mail one-time code (6 digits, 10 min, single use, HMAC-stored). 5 wrong guesses destroy the code and lock the address for 15 min. Deny-by-default allow-list (addresses and/or domains). Identical response for allowed/unknown addresses (no enumeration). |
| Sessions | Random 256-bit token, stored server-side as a hash; `HttpOnly`, `SameSite=Strict`, `Secure` + `__Host-` prefix in production; 12 h absolute / 30 min idle; max 3 per user; revoked on logout. |
| CSRF / origin | Per-session CSRF token on every state-changing call + `Origin` must match host + JSON/multipart content-type enforced. |
| Rate limits | Sliding window, all-or-nothing, in memory: global 120/min/IP · code requests 3/15 min per address and 10/h per IP · code checks 10/15 min per address, 20/15 min per IP · analyses **5/min, 40/h, 150/day per user** and 15/min per IP. Checked **before** the upload is read. IPv6 counted per /64. Returns 429 + `Retry-After`; UI shows a countdown. All configurable (`LIMIT_*`). |
| Uploads | Type from magic bytes (not name/MIME); only JPEG/PNG; 20 MB cap; 128–20000 px per side, 40 MP cap (decompression-bomb guard); polyglots/truncated files rejected; **EXIF/XMP/ICC/comments/PNG text chunks stripped without touching pixels**; images are processed in memory, never written to disk. |
| Internal service | Not published; shared-secret header (constant-time compare); compose puts it on a network with no internet access. Out-of-range scores are never shown. |
| Browser hardening | CSP (`default-src 'self'`, no inline script/style, `frame-ancestors 'none'`), HSTS, `nosniff`, no-referrer, COOP/CORP, `no-store` on all API responses. |
| Audit | Append-only JSONL: pseudonymous user id, hashed IP, event, magnification, coarse result. No e-mail, filename, image or score history. |

## Tests (what was actually run)

* `cd gateway && go test ./...`: 18 tests: limiter, metadata scrubbing (pixels identical), lockout, enumeration,
  CSRF, cross-origin, path traversal, oversize/wrong-type uploads, inference-down, production-config fail-closed.
* `cd inference && pytest`: 12 tests: auth, contract, bad/truncated/bomb images, Keras≡ONNX, preprocessing bit-parity with the training notebook.
* Headless Chromium end-to-end: sign-in, wrong code, upload, result, rate-limit countdown, print layout, dark/mobile, no CSP violations.

**Not verified:** real-model accuracy inside this app (only stand-in weights were available), SMTP delivery,
Docker/compose, TLS, multi-user load, screen-reader behaviour, any third-party security review.

## Known limits

* Single-node: sessions, codes and rate limits live in memory (restart = everyone signs out; several replicas would
  each count separately). Use Redis/Postgres if you need to scale out.
* E-mail OTP is only as strong as the mailbox; add TOTP/WebAuthn if you need more.
* Burned-in text in the pixels (patient names on a slide photo) cannot be removed automatically.
* Score = calibrated-style probability from a model validated on one dataset; shifts in stain/scanner/lab can
  degrade it silently. Treat output as decision *support* for a pathologist, never a result.
* 100x and 400x patients in the training data could not be linked, so cross-magnification generalisation is untested.

## Layout

```
inference/  preprocess.py (training-parity), server.py, tests/        gateway/  *.go, gateway_test.go
web/        React app                                                scripts/  keras_to_onnx.py, make_dummy_models.py, run_dev.sh
deploy/     Dockerfiles, compose, Caddyfile                          models/   put exports here (not shipped)
```
