#!/bin/bash
set -uo pipefail
cd /home/ysc/potree
FILES="07-004 17-004 18-003 19-002 28"
LOG=/home/ysc/potree/training/logs/convert_remaining.log
: > "$LOG"

for f in $FILES; do
  echo "[$(date +%H:%M:%S)] === starting $f === (mem free: $(free -h | awk '/Mem:/{print $7}'))" | tee -a "$LOG"
  python3 training/prepare_dataset.py "/home/ysc/다운로드/${f}.las" /home/ysc/Armstrong/ayg-dna-pcn training/data/lane3d --tile-size 30 --source-id "$f" >> "$LOG" 2>&1
  status=$?
  echo "[$(date +%H:%M:%S)] === $f done, exit=$status === (mem free: $(free -h | awk '/Mem:/{print $7}'))" | tee -a "$LOG"
done

echo "ALL_REMAINING_DONE" | tee -a "$LOG"
