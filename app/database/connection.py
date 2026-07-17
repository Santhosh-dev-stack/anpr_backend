from contextlib import contextmanager
from urllib.parse import urlparse

import psycopg2
from psycopg2.pool import SimpleConnectionPool

from app.config import DATABASE_URL, DB_POOL_MAX_CONN, DB_POOL_MIN_CONN
from app.utils.logger import get_logger

logger = get_logger(__name__)

_pool: SimpleConnectionPool | None = None


def _to_psycopg2_dsn(database_url: str) -> str:
    # Accepts SQLAlchemy-style URLs (postgresql+asyncpg://...) and normalizes
    # them to a plain postgresql:// DSN, which is all psycopg2 understands.
    parsed = urlparse(database_url)
    scheme = parsed.scheme.split("+")[0]
    return parsed._replace(scheme=scheme).geturl()


def get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set")
        dsn = _to_psycopg2_dsn(DATABASE_URL)
        _pool = SimpleConnectionPool(DB_POOL_MIN_CONN, DB_POOL_MAX_CONN, dsn=dsn)
        logger.info("Created PostgreSQL connection pool")
    return _pool


@contextmanager
def get_connection():
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
