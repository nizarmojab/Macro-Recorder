"""
Audit Log — Immutable action audit trail in PostgreSQL.
Every agent action is recorded before and after execution.
"""
import os
from datetime import datetime
from typing import Optional
import psycopg2
from psycopg2.extras import Json

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://agent:agent@localhost:5432/macrodb")


def _get_conn():
    return psycopg2.connect(POSTGRES_URL)


def log_action(
    task: str,
    step: int,
    action: dict,
    status: str,
    error: Optional[str] = None,
    screenshot_before: Optional[str] = None,
    screenshot_after: Optional[str] = None,
):
    """
    Write an immutable audit entry for an agent action.
    Called BEFORE and AFTER execution — never skipped.
    """
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO audit_log
                      (task, step, action, status, error,
                       screenshot_before, screenshot_after, logged_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    task, step, Json(action), status, error,
                    screenshot_before, screenshot_after,
                    datetime.utcnow(),
                ))
    except Exception as e:
        print(f"  [audit] PostgreSQL unavailable, skipping log: {e}")


MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id               BIGSERIAL PRIMARY KEY,
    task             TEXT NOT NULL,
    step             INTEGER NOT NULL,
    action           JSONB NOT NULL,
    status           TEXT NOT NULL,     -- executing | completed | failed
    error            TEXT,
    screenshot_before TEXT,
    screenshot_after  TEXT,
    logged_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS workflows (
    id               BIGSERIAL PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,
    description      TEXT,
    plan             JSONB NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_run_at      TIMESTAMPTZ,
    run_count        INTEGER DEFAULT 0,
    success_count    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS workflow_runs (
    id               BIGSERIAL PRIMARY KEY,
    workflow_id      BIGINT REFERENCES workflows(id),
    status           TEXT NOT NULL,     -- running | completed | failed | waiting_human
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at     TIMESTAMPTZ,
    steps_total      INTEGER,
    steps_completed  INTEGER DEFAULT 0,
    error            TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_task ON audit_log(task);
CREATE INDEX IF NOT EXISTS idx_audit_logged_at ON audit_log(logged_at DESC);
"""


def run_migrations():
    """Run database migrations. Called during setup."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(MIGRATION_SQL)
    print("Database migrations complete")
