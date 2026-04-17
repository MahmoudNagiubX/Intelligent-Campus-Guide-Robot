"""
Navigator storage package.
Public interface for database access and schema management.
"""

from app.storage.db import get_db, close_db
from app.storage.schema import bootstrap_schema, rebuild_fts_indexes, get_table_counts

__all__ = [
    "get_db",
    "close_db",
    "bootstrap_schema",
    "rebuild_fts_indexes",
    "get_table_counts",
]
