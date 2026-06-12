from __future__ import annotations

import os
import pickle
import sqlite3
import subprocess
from pathlib import Path


DB_PATH = Path("demo-users.db")
UPLOAD_ROOT = Path("uploads")


def find_user(username: str) -> list[tuple[int, str, str]]:
    """Demo vulnerability: SQL is built by string concatenation."""
    connection = sqlite3.connect(DB_PATH)
    query = "SELECT id, username, role FROM users WHERE username = '" + username + "'"
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


def export_report(report_name: str) -> str:
    """Demo vulnerability: untrusted input reaches a shell command."""
    command = "tar -czf exports/" + report_name + ".tgz reports/" + report_name
    return subprocess.check_output(command, shell=True, text=True)


def save_upload(filename: str, content: bytes) -> Path:
    """Demo vulnerability: filename is joined without path traversal checks."""
    destination = UPLOAD_ROOT / filename
    destination.write_bytes(content)
    return destination


def load_session(raw_cookie: bytes) -> object:
    """Demo vulnerability: pickle deserializes attacker-controlled bytes."""
    return pickle.loads(raw_cookie)


def reset_password(email: str) -> str:
    """Demo vulnerability: predictable token and secret leakage."""
    token = str(abs(hash(email)) % 1000000)
    smtp_password = os.environ.get("SMTP_PASSWORD", "demo-secret-password")
    return f"token={token}; smtp_password={smtp_password}"
