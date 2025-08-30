import pandas as pd
import numpy as np
import pytest

from database_connectors.classes.database_connector import DatabaseConnector
from database_connectors.classes.database_type import DatabaseType


class FakeConn:
    """Minimal connection stub for constructor paths that don't hit the DB.
    Provides attributes used by methods that might be invoked during init or checks.
    """
    def __init__(self, kind):
        self.kind = kind
        # For PostgreSQL connection checks:
        self.closed = 0
        # psycopg2 extensions uses STATUS_*; emulate "good"
        class Status:
            STATUS_BAD = 0
        self.status = 1
        self._cursors = []

    def cursor(self):
        # Only needed if any method accidentally tries to create a cursor; keep minimal
        class C:
            def execute(self, *a, **k): pass
            def fetchall(self): return []
            def close(self): pass
        c = C()
        self._cursors.append(c)
        return c

    def commit(self): pass
    def rollback(self): pass

    # For MySQL connection checks:
    def ping(self, reconnect=False, attempts=1, delay=0): return True

    # For SQLite connection checks (not used here):
    def execute(self, *a, **k): pass


@pytest.fixture
def patch_mysql_connect(monkeypatch):
    """Fixture for a fake MySQL connection."""
    import mysql.connector as mysql

    # Helper func
    def fake_connect(**kwargs): return FakeConn("mysql")
    
    # Patch for a mysql connection
    monkeypatch.setattr(mysql, "connect", fake_connect)
    
    # Return the helper func
    return fake_connect


@pytest.fixture
def patch_psql_connect(monkeypatch):
    """Fixture for a fake PSQL connection."""
    import psycopg2 as psql

    # Helper func
    def fake_connect(**kwargs): return FakeConn("psql")
        
    # Patch for a PSQL connection
    monkeypatch.setattr(psql, "connect", fake_connect)
    
    # Return the helper func
    return fake_connect


def test_quote_identifier_mysql(patch_mysql_connect):
    """Testing the DatabaseConnector.quote_identifier() for MySQL."""

    # Init a dummy DB connection
    db = DatabaseConnector(
        database_type=DatabaseType.MYSQL,
        host="h",
        username="u",
        password="p",
        database="d",
        enable_logging=False,
    )

    # Safe parts pass through; unsafe get backticked per part
    assert db._quote_identifier("users") == "users"                             # VALID/UNCHANGED
    assert db._quote_identifier("weird-name") == "`weird-name`"                 # ADD BACKTICKS
    assert db._quote_identifier('schema.table') == "schema.table"               # VALID/UNCHANGED
    assert db._quote_identifier('sch.odd name.tbl') == 'sch.`odd name`.tbl'     # ADD BACKTICKS


def test_quote_identifier_postgres(patch_psql_connect):
    """Testing the DatabaseConnector.quote_identifier() for PSQL."""

    # Init a dummy DB connection
    db = DatabaseConnector(
        database_type=DatabaseType.POSTGRESQL,
        host="h",
        username="u",
        password="p",
        database="d",
        enable_logging=False,
    )

    # Verify the PSQL quote identifier 
    assert db._quote_identifier("users") == "users"                         # VALID/UNCHANGED
    assert db._quote_identifier('odd name') == '"odd name"'                 # ADD QUOTES
    assert db._quote_identifier('public."weird"') == 'public."""weird"""'   # ADD QUOTES
    assert db._quote_identifier('a.b.c') == 'a.b.c'                         # VALID/UNCHANGED (PARTS)


def test_split_qualified_no_schema():
    """Testing the DatabaseConnector._split_qualified() for SQLITE."""

    # Use SQLite for minimal init
    db = DatabaseConnector(
        database_type=DatabaseType.SQLITE,
        host="",
        username="",
        password="",
        database=":memory:",
        enable_logging=False,
    )

    # Verify functionality
    assert db._split_qualified("table") == (None, "table")
    assert db._split_qualified("schema.table") == ("schema", "table")
    assert db._split_qualified("db.schema.table") == ("db.schema", "table")


@pytest.mark.parametrize(
    "dbtype,dtype,expected",
    [
        (DatabaseType.POSTGRESQL, np.dtype("int64"), "BIGINT"),
        (DatabaseType.POSTGRESQL, np.dtype("float64"), "DOUBLE PRECISION"),
        (DatabaseType.POSTGRESQL, np.dtype("bool"), "BOOLEAN"),
        (DatabaseType.POSTGRESQL, "datetime64[ns]", "TIMESTAMP"),
        (DatabaseType.POSTGRESQL, "timedelta64[ns]", "INTERVAL"),
        (DatabaseType.POSTGRESQL, np.dtype("object"), "TEXT"),

        (DatabaseType.MYSQL, np.dtype("int64"), "BIGINT"),
        (DatabaseType.MYSQL, np.dtype("float64"), "DOUBLE"),
        (DatabaseType.MYSQL, np.dtype("bool"), "BOOLEAN"),
        (DatabaseType.MYSQL, "datetime64[ns]", "DATETIME"),
        (DatabaseType.MYSQL, "timedelta64[ns]", "TEXT"),
        (DatabaseType.MYSQL, np.dtype("object"), "TEXT"),

        (DatabaseType.SQLITE, np.dtype("int64"), "INTEGER"),
        (DatabaseType.SQLITE, np.dtype("float64"), "REAL"),
        (DatabaseType.SQLITE, np.dtype("bool"), "INTEGER"),
        (DatabaseType.SQLITE, "datetime64[ns]", "TEXT"),
        (DatabaseType.SQLITE, "timedelta64[ns]", "TEXT"),
        (DatabaseType.SQLITE, np.dtype("object"), "TEXT"),
    ],
)


def test_sql_type_for_dtype(dbtype, dtype, expected, patch_mysql_connect, patch_psql_connect):
    """Testing the DatabaseConnector datatype for SQLITE, MySQL, and PSQL connections."""

    # Instantiate appropriate connector (mock out MySQL/PG to avoid real connections)
    if dbtype is DatabaseType.SQLITE:
        db = DatabaseConnector(
            database_type=dbtype,
            host="",
            username="",
            password="",
            database=":memory:",
            enable_logging=False,
        )
    elif dbtype is DatabaseType.MYSQL:
        db = DatabaseConnector(
            database_type=dbtype,
            host="h",
            username="u",
            password="p",
            database="d",
            enable_logging=False,
        )
    else:
        db = DatabaseConnector(
            database_type=dbtype,
            host="h",
            username="u",
            password="p",
            database="d",
            enable_logging=False,
        )

    # Build a one-column DF to extract dtype from pandas pathway where necessary
    if isinstance(dtype, str):
        s = pd.Series(pd.array([], dtype=dtype))
        dt = s.dtype
    else:
        dt = dtype

    assert db._sql_type_for_dtype(dt) == expected


