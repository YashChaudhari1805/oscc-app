"""Cross-platform dev launcher (Windows / macOS / Linux). Starts the inference service and the gateway.

    python scripts/run_dev.py

Sign in at http://127.0.0.1:8080 with any <something>@example.org. The 6-digit code is printed in THIS terminal
(look for 'DEV_ECHO_OTP sign-in code'). Ctrl+C stops everything. Needs: Python deps, Go 1.22+, `npm run build` done."""
import os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = os.environ.get("INTERNAL_TOKEN", "dev-internal-token-change-me-0123456789")
models = Path(os.environ.get("MODEL_DIR", ROOT / "models"))

def die(msg):
    print("\nERROR:", msg, file=sys.stderr); sys.exit(1)

if not any((models / m / "serving_config.json").exists() for m in ("100x", "400x")):
    die(f"No models found in {models}.\n  For a dummy run:  python scripts/make_dummy_onnx.py models\n"
        "  For real models: copy export/100x and export/400x there, then run scripts/keras_to_onnx.py")
if not (ROOT / "web" / "dist" / "index.html").exists():
    die("The web app is not built. Run:  cd web && npm install && npm run build")
go = shutil.which("go")
if not go:
    die("Go was not found on PATH. Install Go 1.22+ (https://go.dev/dl/), then open a NEW terminal.")

env = dict(os.environ, INTERNAL_TOKEN=TOKEN, MODEL_DIR=str(models), APP_ENV="development", DEV_ECHO_OTP="true",
           ALLOWED_DOMAINS=os.environ.get("ALLOWED_DOMAINS", "example.org"), WEB_DIR=str(ROOT / "web" / "dist"),
           AUDIT_LOG=str(ROOT / "audit.jsonl"), PYTHONUNBUFFERED="1")

procs = []
def stop():
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try: p.wait(5)
        except Exception: p.kill()

try:
    print("Starting inference service (loading models, can take a minute)...")
    inf = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "9000",
                            "--log-level", "warning"], cwd=ROOT / "inference", env=env)
    procs.append(inf)
    req = urllib.request.Request("http://127.0.0.1:9000/healthz", headers={"X-Internal-Token": TOKEN})
    for _ in range(240):
        if inf.poll() is not None:
            die("The inference service exited (see the error above). Common causes: missing 'pip install -r inference/requirements.txt', or port 9000 in use.")
        try:
            urllib.request.urlopen(req, timeout=2); break
        except Exception:
            time.sleep(0.5)
    else:
        die("The inference service did not become ready in 2 minutes.")
    print("Inference ready. Starting gateway (first run compiles Go code, ~30 s)...")
    gw = subprocess.Popen([go, "run", "."], cwd=ROOT / "gateway", env=env)
    procs.append(gw)
    time.sleep(3)
    print("\n>>> Open http://127.0.0.1:8080  and sign in as anyone@example.org")
    print(">>> The sign-in code is printed below when you press 'Send code'. Ctrl+C to stop.\n")
    while gw.poll() is None and inf.poll() is None:
        time.sleep(0.5)
    if gw.poll() not in (None, 0):
        die("The gateway exited with an error (see above). If it says 'address already in use', close whatever is using port 8080.")
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stop()
