#!/bin/bash
set -e

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/app"

wait_for_api() {
    local max_attempts="${1:-30}"
    local attempt=1

    echo "Waiting for Datara API (${DATARA_API_BASE_URL}/health)..."

    while [ "$attempt" -le "$max_attempts" ]; do
        if python -c "
import os
import urllib.error
import urllib.request

base = os.environ.get('DATARA_API_BASE_URL', '').rstrip('/')
if not base:
    raise SystemExit(1)
urllib.request.urlopen(base + '/health', timeout=3)
" 2>/dev/null; then
            echo "✓ Datara API is ready"
            return 0
        fi
        echo "  Attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "⚠ Warning: API did not become available in time"
    return 1
}

if [ "${DATARA_WAIT_FOR_API:-0}" = "1" ] && [ -n "${DATARA_API_BASE_URL:-}" ]; then
    wait_for_api || true
fi

exec "$@"
