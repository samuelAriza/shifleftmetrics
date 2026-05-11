#!/usr/bin/env bash
# Upload curated raw datasets into the Databricks UC Volume landing zone.
set -euo pipefail

VOLUME_ROOT="dbfs:/Volumes/workspace/shiftmetrics_bronze/lakehouse_vol/landing"

echo "-> Uploading PROMISE → $VOLUME_ROOT/promise"
databricks fs cp --recursive --overwrite \
  data/raw/promise \
  "$VOLUME_ROOT/promise"

echo ""
echo "-> Uploading Red Hat (250 clean systems) → $VOLUME_ROOT/redhat"
databricks fs cp --recursive --overwrite \
  data/raw/redhat_clean \
  "$VOLUME_ROOT/redhat"

echo ""
echo "════════════════ VOLUME VERIFICATION ════════════════"

echo ""
echo "[*] Top-level landing/:"
databricks fs ls "$VOLUME_ROOT" -o json | python3 -c "
import json, sys
for f in json.load(sys.stdin):
    icon = '[DIR]' if f.get('is_directory') else '[FILE]'
    print(f\"  {icon} {f['name']}\")
"

echo ""
echo "-> promise/ subdirectories (Apache projects):"
databricks fs ls "$VOLUME_ROOT/promise" -o json | python3 -c "
import json, sys
dirs = sorted([f['name'] for f in json.load(sys.stdin) if f.get('is_directory')])
print(f'  {len(dirs)} project dirs: {dirs}')
"

echo ""
echo "▶ Sample: promise/ant/ contents (5 versions):"
databricks fs ls "$VOLUME_ROOT/promise/ant" -o json | python3 -c "
import json, sys
for f in json.load(sys.stdin):
    size_kb = (f.get('file_size', 0) or 0) / 1024
    print(f\"    [FILE] {f['name']:30s}  {size_kb:>8.1f} KB\")
"

echo ""
echo "▶ redhat/ — first 10 of 250 system CSVs:"
databricks fs ls "$VOLUME_ROOT/redhat" -o json | python3 -c "
import json, sys
files = sorted(json.load(sys.stdin), key=lambda x: x['name'])
for f in files[:10]:
    size_kb = (f.get('file_size', 0) or 0) / 1024
    print(f\"    [FILE] {f['name']:30s}  {size_kb:>8.1f} KB\")
print(f'    ... ({len(files)} total files)')
"

echo ""
echo "-> Total objects under landing/ (recursive):"
total=$(databricks fs ls "$VOLUME_ROOT" --recursive -o json 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null) || \
total="unknown (recursive flag may differ)"
echo "    $total"
