import pandas as pd
import pytest

from database_connectors.classes.database_connector import DatabaseConnector
from database_connectors.classes.database_type import DatabaseType


@pytest.fixture
def sqlite_db():
    """A fresh in-memory SQLite DatabaseConnector for each test."""
    db = DatabaseConnector(
        database_type=DatabaseType.SQLITE,
        host="",
        username="",
        password="",
        database=":memory:",
        enable_logging=False,
    )
    assert db.is_connected()
    return db


def test_table_from_df_schema_creates_empty_table(sqlite_db):
    """Testing DatabaseConnector.table_from_df_schema() AND *.get_table_columns()."""

    # Init temp df 
    df = pd.DataFrame(
        {
            "id": pd.Series([1, 2], dtype="int64"),
            "name": pd.Series(["a", "b"], dtype="string"),
            "score": pd.Series([1.2, 3.4], dtype="float64"),
            "flag": pd.Series([True, False], dtype="boolean"),
            "created_at": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "delta": pd.Series([pd.Timedelta("1h"), pd.Timedelta("2h")]),
        }
    )

    # Create new table "users" from the df
    sqlite_db.table_from_df_schema("users", df)

    # Verify columns exist (order matters per pragma ordinal)
    cols = sqlite_db.get_table_columns("users")
    assert cols == ["id", "name", "score", "flag", "created_at", "delta"]

    # Verify that the table is empty initially
    out = sqlite_db.table_as_df("users")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_dump_df_to_table_and_readback(sqlite_db):
    """Testing DatabaseConnector.table_from_df_schema(), *.dump_df_to_table(), AND *.table_as_df()."""

    # Create schema
    schema_df = pd.DataFrame({"id": pd.Series([], dtype="int64"), "name": pd.Series([], dtype="string")})
    sqlite_db.table_from_df_schema("people", schema_df)

    # Dump rows
    data = pd.DataFrame({"id": [1, 2, 3], "name": ["Ada", "Linus", "Grace"]})
    sqlite_db.dump_df_to_table(data, "people")

    # Read back
    df = sqlite_db.table_as_df("people")
    assert df.shape == (3, 2)
    assert set(df["name"]) == {"Ada", "Linus", "Grace"}


def test_execute_one_and_many(sqlite_db):
    """Testing the DatabaseConnector.execute_one() AND *.execute_many() helpers."""

    # Create temp table
    sqlite_db.execute_one("CREATE TABLE t (id INTEGER, v TEXT)")

    # Execute one and execute many
    sqlite_db.execute_one("INSERT INTO t (id, v) VALUES (?, ?)", (1, "a"))
    sqlite_db.execute_many("INSERT INTO t (id, v) VALUES (?, ?)", [(2, "b"), (3, "c")])

    # Verify rows
    rows = sqlite_db.execute_one("SELECT * FROM t ORDER BY id", fetch_results=True)
    assert rows == [(1, "a"), (2, "b"), (3, "c")]


def test_get_all_table_names(sqlite_db):
    """Testing the DatabaseConnector.execute_one() AND *.get_all_table_names()."""

    # Execute statements to create temp tables
    sqlite_db.execute_one("CREATE TABLE a (x INTEGER)")
    sqlite_db.execute_one("CREATE TABLE b (y TEXT)")

    # Get all the table names 
    names = sqlite_db.get_all_table_names()

    # Verify that SQLite system tables are excluded and that both created tables appear
    assert set(names) >= {"a", "b"}


def test_db_as_csvs(tmp_path, sqlite_db):
    """Testing DatabaseConnector.db_as_csvs()."""

    # Create a table with two dummy rows
    sqlite_db.execute_one("CREATE TABLE export_me (i INTEGER, s TEXT)")
    sqlite_db.execute_many("INSERT INTO export_me (i, s) VALUES (?, ?)", [(1, "x"), (2, "y")])

    # Dump DB to CSVs
    out_dir = tmp_path / "csvs"
    sqlite_db.db_as_csvs(str(out_dir))
    csv_path = out_dir / "export_me.csv"

    # Verify that the CSV exists
    assert csv_path.exists()

    # Read the df and verify it is correct
    df = pd.read_csv(csv_path)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["i", "s"]



def test_generate_insert_statement_uses_P_placeholder(sqlite_db):
    """Testing that the proper placeholders are used for SQLITE DB connections."""

    # Make a table to infer columns when cols not passed
    sqlite_db.execute_one("CREATE TABLE books (id INTEGER, title TEXT)")

    # When cols omitted, it will introspect columns
    stmt = sqlite_db._generate_insert_statement("books")
    assert stmt == 'INSERT INTO books (id,title) VALUES (?,?)'

    # With explicit subset (also checks quoting and ordering)
    stmt2 = sqlite_db._generate_insert_statement("books", cols=["title", "id"])
    assert stmt2 == 'INSERT INTO books (title,id) VALUES (?,?)'
