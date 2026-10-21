"""Database connection factory for DuckDB.

Provides a single entry point for connecting to the project database.
Defaults to read-only mode to prevent accidental writes and DuckDB
single-writer lock conflicts.
"""

import logging
from pathlib import Path

import ibis

from gpr_zero.constants import BINARY_MATRIX_TABLE

logger = logging.getLogger(__name__)


def get_connection(
    db_path: str | Path | None = None,
    read_only: bool = True,
) -> ibis.BaseBackend:
    """Return an ibis DuckDB connection.

    Args:
        db_path: Path to the DuckDB file. Defaults to the active project database.
        read_only: Open in read-only mode (default ``True``).

    Returns:
        An ibis DuckDB backend connection.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    if db_path is None:
        from gpr_zero.project import get_project

        db_path = get_project().database
    resolved = Path(db_path)
    if read_only and not resolved.exists():
        raise FileNotFoundError(f"Database not found: {resolved}")

    conn = ibis.duckdb.connect(str(resolved), read_only=read_only)
    logger.info("Connected to %s (read_only=%s)", resolved, read_only)
    return conn


def get_binary_matrix(conn: ibis.BaseBackend) -> ibis.Table:
    """Return the cached binary matrix table from an open connection.

    Args:
        conn: An ibis DuckDB connection that already has the table
              ``gpr_binary_matrix_all_words`` materialized.

    Returns:
        The binary matrix ibis table expression.

    Raises:
        ValueError: If the table does not exist in the connection.
    """
    try:
        return conn.table(BINARY_MATRIX_TABLE)
    except Exception as exc:
        raise ValueError(
            f"Table '{BINARY_MATRIX_TABLE}' not found. "
            "Run the explicit cache-build workflow first."
        ) from exc
