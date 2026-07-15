#!/bin/bash
# Converts every *.las file in LAS_DIR into SemanticKITTI-format training tiles.
# Skips files already converted (marked by a .done_<source_id> file in OUT_DIR),
# so re-running after a crash/OOM only retries what's left - this replaces the
# old convert_remaining.sh hardcoded retry list.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAS_DIR="${1:-$HOME/다운로드}"
HDMAP_DIR="${2:-$HOME/ayg-dna-pcn}"
OUT_DIR="${3:-$SCRIPT_DIR/training/data/lane3d}"

cd "$SCRIPT_DIR"
mkdir -p "$OUT_DIR" training/logs
LOG="training/logs/convert_all.log"
: > "$LOG"

shopt -s nullglob
for las in "$LAS_DIR"/*.las; do
  source_id="$(basename "$las" .las)"
  done_marker="$OUT_DIR/.done_${source_id}"

  if [ -f "$done_marker" ]; then
    echo "[$(date +%H:%M:%S)] === skipping $source_id (already converted) ===" | tee -a "$LOG"
    continue
  fi

  echo "[$(date +%H:%M:%S)] === starting $source_id === (mem free: $(LC_ALL=C free -h | awk '/Mem:/{print $7}'))" | tee -a "$LOG"
  python3 training/prepare_dataset.py "$las" "$HDMAP_DIR" "$OUT_DIR" --tile-size 30 --source-id "$source_id" >> "$LOG" 2>&1
  status=$?
  echo "[$(date +%H:%M:%S)] === $source_id done, exit=$status ===" | tee -a "$LOG"
  [ "$status" -eq 0 ] && touch "$done_marker"
done

echo "ALL_CONVERT_DONE" | tee -a "$LOG"
