"""Small SQL adapter: SQLite snapshot by default, optional DuckDB engine."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Iterator
from .config import DB_PATH

class Database:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = 'duckdb' if self.path.suffix == '.duckdb' else 'sqlite'
        if self.engine == 'duckdb':
            try:
                import duckdb
            except ImportError as e:
                raise RuntimeError('DuckDB database requested: pip install duckdb') from e
            self.conn = duckdb.connect(str(self.path))
        else:
            self.conn = sqlite3.connect(str(self.path), timeout=45)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute('PRAGMA foreign_keys=ON')
            self.conn.execute('PRAGMA journal_mode=WAL')

    def execute(self, sql: str, args: tuple | list = ()) -> Any:
        return self.conn.execute(sql, args)

    def many(self, sql: str, rows: list[tuple]) -> None:
        if rows:
            self.conn.executemany(sql, rows)

    def query(self, sql: str, args: tuple | list = ()) -> list[dict]:
        cursor = self.conn.execute(sql, args)
        columns = [col[0] for col in cursor.description or []]
        return [dict(zip(columns, tuple(row))) for row in cursor.fetchall()]

    def one(self, sql: str, args: tuple | list = ()) -> dict | None:
        rows = self.query(sql, args)
        return rows[0] if rows else None

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> 'Database':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        self.close()


def require_prepared(db: Database) -> None:
    try:
        db.one('SELECT COUNT(*) AS n FROM orders')
        db.one('SELECT COUNT(*) AS n FROM sample_reviews')
    except Exception as e:
        raise RuntimeError('Database not prepared. Run: python -m revenue_risk.cli prepare') from e
