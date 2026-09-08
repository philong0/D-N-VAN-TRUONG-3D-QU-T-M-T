#!/usr/bin/env bash
# Starts ai-engine (FastAPI, :8001) first and waits for it to actually be
# ready (measured cold boot ~6.2s — loading YuNet/PIPNet/torch weights)
# before starting Next.js dev — avoids the real race where Canvas3D's GNM
# multi-view fetch hits ai-engine before it can accept connections and
# permanently falls back to the 3DDFA_V2 (frontal-only) pipeline for that
# page load. Both existing pipelines (D3/D4.5 GNM fit, 3DDFA_V2 fallback)
# are unchanged by this script — it only sequences process startup.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

AI_ENGINE_PORT="${AI_ENGINE_PORT:-8001}"
AI_ENGINE_HEALTH_URL="http://localhost:${AI_ENGINE_PORT}/test"
READY_TIMEOUT_S=30

AI_PID=""
cleanup() {
  if [ -n "$AI_PID" ] && kill -0 "$AI_PID" 2>/dev/null; then
    echo "[dev-all] stopping ai-engine (pid $AI_PID)..."
    kill "$AI_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if curl -sf --max-time 1 "$AI_ENGINE_HEALTH_URL" >/dev/null 2>&1; then
  echo "[dev-all] ai-engine already running on :$AI_ENGINE_PORT — reusing it."
else
  echo "[dev-all] starting ai-engine on :$AI_ENGINE_PORT ..."
  (cd ai-engine && source venv/bin/activate && exec python main.py) &
  AI_PID=$!

  echo "[dev-all] waiting for ai-engine to become ready (up to ${READY_TIMEOUT_S}s)..."
  ready=0
  for i in $(seq 1 "$READY_TIMEOUT_S"); do
    if curl -sf --max-time 1 "$AI_ENGINE_HEALTH_URL" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [ "$ready" != "1" ]; then
    echo "[dev-all] ERROR: ai-engine did not become ready within ${READY_TIMEOUT_S}s — check ai-engine's own output above." >&2
    exit 1
  fi
  echo "[dev-all] ai-engine ready."
fi

echo "[dev-all] starting Next.js dev..."
npm run dev
