"""
Navigator storage package.
Public interface for database access and schema management.
"""

from app.storage.db import close_db, get_db
from app.storage.schema import bootstrap_schema, get_table_counts, rebuild_fts, rebuild_fts_indexes

__all__ = [
    "get_db",
    "close_db",
    "bootstrap_schema",
    "rebuild_fts",
    "rebuild_fts_indexes",
    "get_table_counts",
]
