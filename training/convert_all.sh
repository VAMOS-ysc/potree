#!/bin/bash
set -uo pipefail
cd /home/ysc/potree
FILES="02-003 03-005 03-extra 04 05-004 06-005 07-004 08-001 09-002 10-003 17-004 18-003 19-002 28"
LOG=/home/ysc/potree/training/logs/convert_all.log
: > "$LOG"

for f in $FILES; do
  echo "[$(date +%H:%M:%S)] === starting $f ===" | tee -a "$LOG"
  python3 training/prepare_dataset.py "/home/ysc/다운로드/${f}.las" /home/ysc/Armstrong/ayg-dna-pcn training/data/lane3d --tile-size 30 --source-id "$f" >> "$LOG" 2>&1
  status=$?
  echo "[$(date +%H:%M:%S)] === $f done, exit=$status ===" | tee -a "$LOG"
done

echo "ALL_CONVERT_DONE" | tee -a "$LOG"
