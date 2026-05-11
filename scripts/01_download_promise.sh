#!/usr/bin/env bash
# Download the PROMISE Defect Dataset from the feiwww/PROMISE-backup GitHub mirror.
# Idempotent: skips files already present.
set -euo pipefail

DEST="data/raw/promise"
BASE="https://raw.githubusercontent.com/feiwww/PROMISE-backup/master/bug-data"

# Verified against repo on 2026-05-10 — DO NOT ASSUME, this list is canonical.
declare -A VERSIONS=(
  [ant]="1.3 1.4 1.5 1.6 1.7"
  [camel]="1.0 1.2 1.4 1.6"
  [ivy]="1.0 1.1 1.2"
  [jedit]="3.2 4.0 4.1 4.2 4.3"
  [log4j]="1.0 1.1 1.2"
  [lucene]="2.0 2.2 2.4"
  [poi]="1.5 2.0 2.5 3.0"
  [synapse]="1.0 1.1 1.2"
  [velocity]="1.4 1.5 1.6"
  [xalan]="2.4 2.5 2.6 2.7"
  [xerces]="1.1 1.2 1.3 1.4.4"
)
PROJECTS=(ant camel ivy jedit log4j lucene poi synapse velocity xalan xerces)

mkdir -p "$DEST"
ok=0
skipped=0
failed=0
expected=0

for proj in "${PROJECTS[@]}"; do
  mkdir -p "$DEST/$proj"
  for ver in ${VERSIONS[$proj]}; do
    expected=$((expected+1))
    url="$BASE/$proj/${proj}-${ver}.csv"
    out="$DEST/$proj/${proj}-${ver}.csv"
    if [[ -s "$out" ]]; then
      skipped=$((skipped+1))
      continue
    fi
    if curl -sfL --max-time 60 "$url" -o "$out"; then
      ok=$((ok+1))
      printf "%-20s  %s\n" "$proj-$ver" "$(wc -c <"$out") bytes"
    else
      failed=$((failed+1))
      rm -f "$out"
      echo "FAILED: $url" >&2
    fi
  done
done

echo "  Expected: $expected   Downloaded: $ok   Skipped: $skipped   Failed: $failed"

# Integrity check: every file should have ≥ 2 lines (header + data) and the expected schema
echo "-> Schema integrity check (expecting 22-column header)"

bad=0
for f in $(find "$DEST" -name "*.csv"); do
  cols=$(head -1 "$f" | tr ',' '\n' | wc -l)
  rows=$(($(wc -l <"$f") - 1))
  if [[ $cols -ne 22 || $rows -lt 1 ]]; then
    echo "$f  cols=$cols rows=$rows"
    bad=$((bad+1))
  fi
done
[[ $bad -eq 0 ]] && echo "All $ok files have 22-column schema and ≥1 data row"

echo ""
total_rows=$(find "$DEST" -name "*.csv" -exec tail -n +2 {} \; | wc -l)
echo "  Total data rows across all PROMISE files: $total_rows"
du -sh "$DEST"

[[ $failed -gt 0 ]] && exit 1
