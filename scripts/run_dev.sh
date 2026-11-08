#!/usr/bin/env bash
# Local development stack: inference (9000) + gateway (8080). Codes are printed in the gateway log.
# Usage: scripts/run_dev.sh   (needs: python deps, a built gateway binary or Go, `npm run build` in web/)
set -euo pipefail
cd "$(dirname "$0")/.."
export INTERNAL_TOKEN="${INTERNAL_TOKEN:-dev-internal-token-change-me-0123456789}"
export MODEL_DIR="${MODEL_DIR:-$PWD/models}"
(cd inference && python -m uvicorn server:app --host 127.0.0.1 --port 9000 --log-level warning) &
INF=$!
trap 'kill $INF 2>/dev/null || true' EXIT
export APP_ENV=development DEV_ECHO_OTP=true ALLOWED_DOMAINS="${ALLOWED_DOMAINS:-example.org}" ALLOWED_EMAILS="${ALLOWED_EMAILS:-}"
export WEB_DIR="$PWD/web/dist" AUDIT_LOG="${AUDIT_LOG:-$PWD/audit.jsonl}"
cd gateway && go run .
