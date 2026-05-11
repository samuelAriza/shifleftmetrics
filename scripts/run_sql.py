#!/usr/bin/env python3
"""Execute SQL statements against a Databricks SQL Warehouse via the Statements API."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WAREHOUSE_ID = os.environ.get("WAREHOUSE_ID")
if not WAREHOUSE_ID:
    sys.exit("ERROR: env var WAREHOUSE_ID not set")


def dbx_api(method: str, path: str, payload: dict | None = None) -> dict:
    cmd = ["databricks", "api", method, path]
    if payload is not None:
        cmd += ["--json", json.dumps(payload)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.exit(f"CLI error ({method} {path}): {proc.stderr.strip()}")
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


def submit(stmt: str) -> dict:
    return dbx_api("post", "/api/2.0/sql/statements", {
        "statement": stmt,
        "warehouse_id": WAREHOUSE_ID,
        "wait_timeout": "30s",
        "on_wait_timeout": "CONTINUE",
    })


def wait_until_done(statement_id: str, timeout_s: int = 180) -> tuple[str, str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = dbx_api("get", f"/api/2.0/sql/statements/{statement_id}")
        state = r.get("status", {}).get("state", "?")
        if state in ("SUCCEEDED", "FAILED", "CANCELED", "CLOSED"):
            err = r.get("status", {}).get("error", {}).get("message", "")
            return state, err
        time.sleep(2)
    return "TIMEOUT", "exceeded local timeout"


def split_statements(sql: str) -> list[str]:
    no_comments = "\n".join(
        ln for ln in sql.splitlines() if not ln.strip().startswith("--")
    )
    return [s.strip() for s in no_comments.split(";") if s.strip()]


def main(sql_file: str) -> None:
    text = Path(sql_file).read_text()
    stmts = split_statements(text)
    print(f"- Running {len(stmts)} statement(s) from {sql_file}")
    failures = 0
    for i, s in enumerate(stmts, 1):
        preview = re.sub(r"\s+", " ", s)[:90]
        print(f"  [{i}/{len(stmts)}] {preview}...")
        r = submit(s)
        state = r.get("status", {}).get("state", "?")
        sid = r.get("statement_id", "")
        if state in ("PENDING", "RUNNING"):
            state, err = wait_until_done(sid)
        else:
            err = r.get("status", {}).get("error", {}).get("message", "")
        if state == "SUCCEEDED":
            print(f"        -> SUCCEEDED")
        else:
            print(f"        -> {state}: {err}", file=sys.stderr)
            failures += 1
    if failures:
        sys.exit(f"{failures} statement(s) failed")
    print("All statements SUCCEEDED")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: run_sql.py <path/to/file.sql>")
    main(sys.argv[1])
