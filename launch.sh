#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ZGX_PYTHON:-$APP_DIR/.venv/bin/python}"
PORT="${ZGX_OBSERVATORY_PORT:-8120}"
PIDFILE="$APP_DIR/data/server.pid"
LOG="$APP_DIR/data/server.log"
mkdir -p "$APP_DIR/data"
cd "$APP_DIR"

# Clean up a stale PID file without killing unrelated processes.
if [[ -f "$PIDFILE" ]]; then
  PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    if curl -fsS "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1; then
      echo "Research Observatory V4 already running (PID $PID)."
    else
      echo "Stale/unhealthy server PID detected; restarting."
      kill "$PID" 2>/dev/null || true
      rm -f "$PIDFILE"
    fi
  else
    rm -f "$PIDFILE"
  fi
fi

if [[ ! -f "$PIDFILE" ]]; then
  nohup "$PY" -m uvicorn app:app --host 127.0.0.1 --port "$PORT" --no-access-log >"$LOG" 2>&1 &
  echo $! > "$PIDFILE"
fi

for i in {1..45}; do
  curl -fsS "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1 && break
  sleep 1
done
if ! curl -fsS "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1; then
  echo "ERROR: Research Observatory did not become ready."
  echo "Log: $LOG"
  tail -60 "$LOG" || true
  exit 1
fi

URL="http://127.0.0.1:$PORT"
CHROMIUM="$(command -v chromium || command -v chromium-browser || true)"
if [[ -n "$CHROMIUM" ]]; then
  mkdir -p "$APP_DIR/data/chromium-profile"
  nohup "$CHROMIUM" --app="$URL" --user-data-dir="$APP_DIR/data/chromium-profile" --no-first-run --no-default-browser-check --password-store=basic --disable-session-crashed-bubble --disable-infobars >/dev/null 2>&1 &
  echo "✓ Opened Research Observatory in Chromium app mode."
else
  echo "Open in a browser: $URL"
fi
echo "✓ HP ZGX Research Observatory V4: $URL"
echo "Log: $LOG"
