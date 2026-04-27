#!/bin/bash
set -e

export PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"

# Optional: wait for an external HTTP dependency (not used for GPU-only deployments).
if [ "${GPU_WAIT_FOR_HTTP:-0}" = "1" ] && [ -n "${GPU_WAIT_URL:-}" ]; then
  echo "Waiting for ${GPU_WAIT_URL}..."
  python - <<'PY' || true
import os, time, urllib.request
url = os.environ.get("GPU_WAIT_URL", "").rstrip("/")
if not url:
    raise SystemExit(0)
for i in range(60):
    try:
        urllib.request.urlopen(url, timeout=3)
        print("ready")
        raise SystemExit(0)
    except Exception:
        time.sleep(2)
print("timeout waiting for URL")
PY
fi

exec "$@"
