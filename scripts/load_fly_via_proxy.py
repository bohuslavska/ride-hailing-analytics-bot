"""
Load local parquet into the Fly Postgres cluster through `fly proxy`.

Prerequisites (separate terminal):
    fly proxy 5433:5432 -a ride-hailing-db

Usage:
    .venv/bin/python scripts/load_fly_via_proxy.py

Reads DATABASE_URL from the running app via `fly ssh`, rewrites the host to
localhost:5433, and never prints the password.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.load_to_postgres import load  # noqa: E402


def _pick_machine_id() -> str:
    listed = subprocess.run(
        ["fly", "machines", "list", "-a", "ride-hailing-analytics", "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if listed.returncode != 0:
        raise SystemExit(
            "Could not list Fly machines.\n"
            f"stderr: {listed.stderr.strip() or '(empty)'}"
        )
    import json

    machines = json.loads(listed.stdout or "[]")
    for machine in machines:
        if machine.get("state") == "started":
            return str(machine["id"])
    raise SystemExit("No started Fly machine found for ride-hailing-analytics.")


def _database_url_from_app() -> str:
    # `fly ssh` sometimes times out on WireGuard; `fly machine exec` is enough
    # to read the attached DATABASE_URL secret.
    machine_id = _pick_machine_id()
    result = subprocess.run(
        [
            "fly",
            "machine",
            "exec",
            machine_id,
            "-a",
            "ride-hailing-analytics",
            "printenv DATABASE_URL",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Could not read DATABASE_URL from the Fly app.\n"
            f"stderr: {result.stderr.strip() or '(empty)'}"
        )

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("postgres"):
            return line

    raise SystemExit(
        "fly machine exec returned no postgres URL. Is the app attached to Postgres?"
    )


def _via_local_proxy(url: str, host: str = "localhost", port: int = 5433) -> str:
    parsed = urlparse(url)
    user = parsed.username or ""
    password = parsed.password or ""
    netloc = user
    if password:
        netloc += f":{password}"
    netloc += f"@{host}:{port}"
    path = parsed.path or "/ride_hailing_analytics"
    # Disable TLS for the local proxy hop; Fly's URL often has sslmode=disable.
    query = parsed.query or "sslmode=disable"
    if "sslmode=" not in query:
        query = f"{query}&sslmode=disable" if query else "sslmode=disable"
    return urlunparse(("postgresql+psycopg", netloc, path, "", query, ""))


def main() -> int:
    source = _database_url_from_app()
    proxy_url = _via_local_proxy(source)
    redacted = re.sub(r":([^:@/]+)@", ":***@", proxy_url)
    print(f"loading through {redacted}")

    load(
        ROOT / "data",
        proxy_url,
        ROOT / "sql" / "schema.sql",
        ROOT / "sql" / "indexes.sql",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
