#!/usr/bin/env python3
"""Inspect a Databricks Volume path with correct size formatting (CLI v0.272+)."""
import json, subprocess, sys

def ls(path):
    r = subprocess.run(
        ["databricks", "fs", "ls", path, "-o", "json"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(r.stdout) if r.stdout.strip() else []

def fmt_size(n):
    n = n or 0
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:>7.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

def show(path, limit=None):
    items = ls(path)
    items.sort(key=lambda x: (not x.get("is_directory"), x["name"]))
    print(f"\n[*] {path}")
    print(f"  ({len(items)} entries)")
    shown = items if limit is None else items[:limit]
    for it in shown:
        icon = "[DIR]" if it.get("is_directory") else "[FILE]"
        size = "" if it.get("is_directory") else fmt_size(it.get("size", 0))
        print(f"  {icon} {it['name']:<35s}  {size}")
    if limit and len(items) > limit:
        print(f"  ... {len(items)-limit} more")
    return items

if __name__ == "__main__":
    for p in sys.argv[1:]:
        show(p, limit=10)
