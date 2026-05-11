#!/usr/bin/env bash
# Download Red Hat Public Jira 2001-2024 dataset from Zenodo record 14558684.
set -euo pipefail

DEST="data/raw/redhat"
RECORD_ID="14558684"
ZENODO_API="https://zenodo.org/api/records/${RECORD_ID}"

mkdir -p "$DEST"

echo "-> Querying Zenodo record ${RECORD_ID}..."
META=$(curl -sfL --max-time 30 "$ZENODO_API")
echo "$META" | python3 -c "
import json, sys
r = json.load(sys.stdin)
print(f\"  Title: {r.get('metadata',{}).get('title','?')[:90]}\")
print(f\"  License: {r.get('metadata',{}).get('license',{}).get('id','?')}\")
print(f\"  Files available:\")
for f in r.get('files', []):
    size_mb = f['size'] / (1024*1024)
    print(f\"    {size_mb:>8.1f} MB  {f['key']}\")
"

# Download every file in the record (record has 1 main CSV; safe loop anyway)
echo ""
echo "Downloading..."
echo "$META" | python3 -c "
import json, sys
r = json.load(sys.stdin)
for f in r.get('files', []):
    print(f\"{f['links']['self']}\t{f['key']}\")
" | while IFS=$'\t' read -r url key; do
  out="$DEST/$key"
  if [[ -s "$out" ]]; then
    echo "skip (exists): $key"
    continue
  fi
  echo "$key"
  curl -fL --max-time 1200 "$url" -o "$out" \
    --progress-bar
done

echo ""
ls -lh "$DEST"

# Quick sanity check on first downloaded file
first=$(find "$DEST" -type f | head -1)
if [[ -n "$first" ]]; then
  echo ""
  echo "-> Sanity check on: $(basename "$first")"
  echo "  Type: $(file -b "$first")"
  echo "  Size: $(du -h "$first" | cut -f1)"
  if [[ "$first" == *.csv ]]; then
    echo "  Header (first line):"
    head -1 "$first" | head -c 300
    echo ""
    echo "  Total rows: $(wc -l <"$first")"
  fi
fi
