#!/usr/bin/env bash
# Start a local, loopback-only edge/cloud simulation for Integration validation.
# It intentionally does not represent an externally deployed cloud service.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$ROOT/../llama.cpp/build-wsl/bin/llama-server}"
EDGE_MODEL="${EDGE_MODEL:-$ROOT/../competition-full/hhk/models/Qwen3-1.7B-IQ4_XS.gguf}"
CLOUD_MODEL="${CLOUD_MODEL:-$ROOT/../competition-full/hhk/models/Qwen3-1.7B-GGUF/Qwen3-1.7B-BF16.gguf}"
THREADS="${LLAMA_THREADS:-8}"
CONTEXT_SIZE="${LLAMA_CONTEXT_SIZE:-512}"
LOG_DIR="${LOG_DIR:-$ROOT/integration/outputs/local_services}"

for required in "$LLAMA_SERVER" "$EDGE_MODEL" "$CLOUD_MODEL"; do
    if [[ ! -f "$required" ]]; then
        echo "required file is missing: $required" >&2
        exit 1
    fi
done

if ss -ltnH | grep -qE ':(8080|8081)([[:space:]]|$)'; then
    echo "port 8080 or 8081 is already in use; refusing to replace an existing service" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

nohup "$LLAMA_SERVER" --model "$EDGE_MODEL" --host 127.0.0.1 --port 8080 \
    -c "$CONTEXT_SIZE" -t "$THREADS" -np 1 \
    >"$LOG_DIR/edge-server.log" 2>&1 &
edge_pid=$!

nohup "$LLAMA_SERVER" --model "$CLOUD_MODEL" --host 127.0.0.1 --port 8081 \
    -c "$CONTEXT_SIZE" -t "$THREADS" -np 1 \
    >"$LOG_DIR/cloud-server.log" 2>&1 &
cloud_pid=$!

printf '%s\n' "$edge_pid" > "$LOG_DIR/edge-server.pid"
printf '%s\n' "$cloud_pid" > "$LOG_DIR/cloud-server.pid"

echo "edge local simulation:  http://127.0.0.1:8080 (PID $edge_pid)"
echo "cloud local simulation: http://127.0.0.1:8081 (PID $cloud_pid)"
echo "logs: $LOG_DIR"
