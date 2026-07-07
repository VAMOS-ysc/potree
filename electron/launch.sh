#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PATH="/home/ysc/miniconda3/bin:$PATH"
exec npm run desktop
