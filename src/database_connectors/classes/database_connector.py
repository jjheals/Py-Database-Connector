# Standard imports
import re 
import logging 
import os 
from typing import Iterable, Optional
import pandas as pd 
import numpy as np 

from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
    is_string_dtype,
    is_object_dtype,
)

# Imports for DB drivers
import mysql.connector as mysql
import psycopg2 as psql
import psycopg2.extensions as _psql_ext
import sqlite3 as sqlite

from mysql.connector import MySQLConnection 
from psycopg2.extensions import connection as PSQLConnection
from sqlite3 import Connection as SQLiteConnection

from mysql.connector.cursor import MySQLCursor
from psycopg2.extensions import cursor as PSQLCursor
from sqlite3 import Cursor as SQLiteCursor

# Custom utils and objs 
from ..utils.general import setup_logger
from ..exceptions import DatabaseNotConnected, TableDoesNotExist, DatabaseTypeNotSupported
from .db_cursor import DBCursor
from .database_type import DatabaseType


# Define a regex for preventing against injection
_SAFE_IDENT:re.Pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# DatabaseConnector class definition
class DatabaseConnector(object): 

    database_type:DatabaseType                              # The DatabaseType for this instance
    cxn:MySQLConnection|PSQLConnection|SQLiteConnection     # The database connection object
    enable_logging:bool                                     # Optional - specify whether to enable logging for this instance; defaults to True
    logger:logging.Logger                                   # Logger for debug/info/etc
    P:str                                                   # The placeholder for this DB type (%s or ?)


    def __init__(
            self,
            database_type:DatabaseType, 
            host:str,  
            username:str, 
            password:str, 
            *, 
            port:int|None=None,
            database:str|None=None, 
            enable_logging:bool=True, 
            log_file_path:str='./database_connection.log', 
            logger_name:str='database_connection_logger', 
            logger_min_level:int=logging.DEBUG,
            logger_format:str="%(asctime)s - %(levelname)s: %(message)s"
        ): 
        
        # Set the base attributes
        self.database_type = database_type
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.enable_logging = enable_logging

        # Setup logging if configured
        if enable_logging:

            # Init a logger
            self.logger = setup_logger(
                log_file_path=log_file_path,
                logger_name=logger_name,
                min_level=logger_min_level,
                log_format=logger_format,
            )

        # Set connection based on type
        try: 
            match database_type: 

                # MYSQL DATABASE
                case DatabaseType.MYSQL:

                    # Set port if None is given
                    if port is None: port = 3306

                    # Set P 
                    self.P = '%s'

                    # Connect
                    self.cxn = mysql.connect(
                        database=database,
                        host=host,
                        user=username,
                        password=password
                    )

                # POSTGRESQL DATABASE
                case DatabaseType.POSTGRESQL:

                    # Set port if None is given
                    if port is None: port = 5432

                    # Set P
                    self.P = '%s'

                    # Connect
                    self.cxn = psql.connect(
                        dbname=database,
                        host=host,
                        port=port,
                        user=username,
                        password=password
                    )

                # SQLITE DATABASE
                case DatabaseType.SQLITE:

                    # Set P
                    self.P = '?'

                    if database is None or not database:
                        raise ValueError("For SQLite, 'database' must be a file path or ':memory:'.")
                    
                    # Use the given filepath to connect 
                    self.cxn = sqlite.connect(database)

                # UNSUPPORTED 
                case _: 
                    raise ValueError(f'Given [database_type] of "{database_type}" is not supported.')

        # Handle exceptions
        except Exception as e: 
            self.log_error('__init__()', e)
            self.cxn = None


    # ---- Helper functions for standardizing logging ---- #
    def _log(
        self,
        level:int,
        fmt:str,
        *args,
        exc:BaseException|None=None,
        stacklevel:int=2,
    ) -> None:
        """Helper func to standardize logging format (or do nothing if not [self.enable_logging or not self.logger). 
        Log format is: "[calling_function]: [message|Exception]" """

        # Check if enable logging is True
        if not getattr(self, "enable_logging", False): return

        # Make sure self.Logger is not None
        logger:logging.Logger = getattr(self, "logger", None)
        if logger is None: return

        # Write to the log
        logger.log(level, fmt, *args, exc_info=exc, stacklevel=stacklevel)


    def log_debug(self, calling_func:str, message:str, stacklevel:int=2) -> None: 
        """Logs a DEBUG message."""
        self._log(logging.DEBUG, "%s: %s", calling_func, message, stacklevel=stacklevel)


    def log_warning(self, calling_func:str, message:str, stacklevel:int=2) -> None: 
        """Logs a WARNING message."""
        self._log(logging.WARNING, "%s error (non-critical): %s", calling_func, message, stacklevel=stacklevel)


    def log_error(self, calling_func:str, exception:Exception, stacklevel:int=2) -> None: 
        """Logs an ERROR message."""
        self._log(logging.ERROR, "%s failed: %s - %s", calling_func, type(exception).__name__, exception, exc=exception, stacklevel=stacklevel)
    

    # ---- Functions for checking if the database connection is running and healthy ---- #
    def _ensure_cxn(self) -> None: 
        """Raises a DatabaseNotConnected exception if the DB is not connected."""   
        if not self._check_connection():
            self.log_error('_ensure_cxn()', DatabaseNotConnected())
            raise DatabaseNotConnected()
    

    def _check_connection(self) -> bool:
        """Returns True if the connection is running and is healthy, False otherwise."""

        # Base case: self.cxn is None
        if self.cxn is None: return False

        # Check based on db type
        match self.database_type:
            case DatabaseType.MYSQL: 
                try:
                    self.cxn.ping(reconnect=False, attempts=1, delay=0)
                    return True
                except Exception:
                    pass
            case DatabaseType.POSTGRESQL: 
                if getattr(self.cxn, "closed", 1) == 0 and getattr(self.cxn, "status", _psql_ext.STATUS_BAD) != _psql_ext.STATUS_BAD: 
                    return True
            case DatabaseType.SQLITE: 
                try: 
                    self.cxn.execute('SELECT 1;')
                    return True
                except sqlite.ProgrammingError: 
                    pass

        # Not connected if we make it here
        return False


    def is_connected(self) -> bool: 
        """Public method for checking if the DB is connected and the connection is healthy (does not raise Exceptions)."""
        return self._check_connection()
    

    # ---- Methods for validating and cleaning inputs and identifiers ---- #
    def _quote_identifier(self, identifier: str) -> str:
        """Safely quote an identifier or dotted path (db.schema.table) per backend. Safe parts (regex) are left unquoted.
            - Postgres/SQLite: "part" with internal " doubled -> ""
            - MySQL: `part` with internal backticks doubled -> ``
        """

        # Clean the identifier of whitespace
        name:str = (identifier or "").strip()

        # Make sure the identifier wasn't empty or just whitespace
        if not name:
            raise ValueError("Empty identifier")

        # Strip down to parts
        parts = [p.strip() for p in name.split(".")]
        
        # Check that parts are all valid
        if any(p == "" for p in parts):
            raise ValueError(f"Invalid identifier: {identifier!r}")

        # Quote each part individually based on DB type
        quoted_parts: list[str] = []

        match self.database_type:

            # POSTGRESQL OR SQLITE
            case DatabaseType.POSTGRESQL | DatabaseType.SQLITE:
                for p in parts:
                    if _SAFE_IDENT.fullmatch(p):
                        quoted_parts.append(p)
                    else:
                        quoted_parts.append(f"\"{p.replace('\"', '\"\"')}\"")
                return ".".join(quoted_parts)

            # MYSQL
            case DatabaseType.MYSQL:
                for p in parts:
                    if _SAFE_IDENT.fullmatch(p):
                        quoted_parts.append(p)
                    else:
                        quoted_parts.append(f"`{p.replace('`', '``')}`")
                return ".".join(quoted_parts)

            # UNSUPPORTED
            case _:
                raise ValueError(f"Unsupported database type: {self.database_type}")
        

    def _split_qualified(self, name: str) -> tuple[str|None, str]:
        """Split 'schema.table' (or just 'table') into (schema, table). Returns (None, table) if no schema is provided."""

        # Split into parts (on '.')
        parts:list[str] = (name or "").strip().split(".")
        
        # If no schema, return (None, [table name])
        if len(parts) == 1: return None, parts[0]
        
        # If two parts, return ([schema], [table name])
        if len(parts) == 2: return parts[0], parts[1]
        
        # If more than 2 parts, treat last part as table, join the rest as schema, and return ([schema], [table name])
        return ".".join(parts[:-1]), parts[-1]


    def _sql_type_for_dtype(self, dtype:pd.api.extensions.ExtensionDtype|np.dtype) -> str:
        """Map a pandas dtype to a backend SQL type.
        
        NOTE:
            - Uses BIGINT for safety over INT (incase of larger integers)
            - For PSQL, technically float64 should convert to DOUBLE PRECISION, and float32 to REAL - this method uses DOUBLE PRECISION for both

        """
        # Normalize dtype
        try: dt = pd.api.types.pandas_dtype(dtype)
        except Exception: dt = np.dtype("object")

        # Find datatype based on this instance's database type
        match self.database_type: 

            # POSTGRESQL
            case DatabaseType.POSTGRESQL:
                if is_integer_dtype(dt): return "BIGINT"
                if is_float_dtype(dt): return "DOUBLE PRECISION"
                if is_bool_dtype(dt): return "BOOLEAN"
                if is_datetime64_any_dtype(dt): return "TIMESTAMP"
                if is_timedelta64_dtype(dt): return "INTERVAL"
                if is_string_dtype(dt) or is_object_dtype(dt): return "TEXT"
                
                # Catchall - bytes type
                return "BYTEA"

        # MYSQL
            case DatabaseType.MYSQL:
                if is_integer_dtype(dt): return "BIGINT"
                if is_float_dtype(dt): return "DOUBLE"
                if is_bool_dtype(dt): return "BOOLEAN"                          # NOTE: alias of TINYINT(1)
                if is_datetime64_any_dtype(dt): return "DATETIME"   
                if is_timedelta64_dtype(dt): return "TEXT"                      # NOTE: MySQL has no INTERVAL column type, so use TEXT
                if is_string_dtype(dt) or is_object_dtype(dt): return "TEXT"    # NOTE: TEXT instead of VARCHAR for variable lengths

                # Catchall - blob type
                return "BLOB"

            # SQLITE (affinity types)
            case DatabaseType.SQLITE:
                if is_integer_dtype(dt): return "INTEGER"
                if is_float_dtype(dt): return "REAL"
                if is_bool_dtype(dt): return "INTEGER"                          # NOTE: SQLite booleans are typically 0/1
                if is_datetime64_any_dtype(dt): return "TEXT"                   # NOTE: use TEXT for ISO-8601 text
                if is_timedelta64_dtype(dt): return "TEXT"                      # NOTE: use TEXT for intervals
                if is_string_dtype(dt) or is_object_dtype(dt): return "TEXT"    

                # Catchall - blob type
                return "BLOB"

            # UNSUPPORTED
            case _: 
                raise DatabaseTypeNotSupported(self.database_type)


    def _commit(self, rollback_on_error:bool=True) -> None: 
        """Helper that commits changes to self.cxn and logs any errors if they occur."""
        try: self.cxn.commit()
        except Exception as e: 
            self.log_warning('_commit()', f'Error when committing changes: {e.__class__.__name__} - {e}')
            if rollback_on_error: 
                self.cxn.rollback()
                self.log_warning('_commit()', 'Rolled back changes.')
            

    def _generate_insert_statement(self, raw_table_name:str, cols:list[str]|None=None) -> str: 
        """Cleans the given table name and generates an INSERT statement for the table, with placeholders depending on
        this instance's self.P. If cols are passed, those columns are used; otherwise the entire set of columns for
        the give table are used."""

        # Get the table columns 
        if not cols: 
            cols = self.get_table_columns(raw_table_name)

        # Validate cols
        if not cols: 
            self.log_error('_generate_insert_statement()', ValueError(f'No columns found for table: "{raw_table_name}"'))
            return 
        
        # Clean table name
        table_id:str = self._quote_identifier(raw_table_name)

        # Construct the query parts
        cols_str:str = ','.join([self._quote_identifier(c) for c in cols])
        placeholders_str:str = ','.join([self.P for _ in cols])

        # Construct and return the query
        return f'INSERT INTO {table_id} ({cols_str}) VALUES ({placeholders_str})'


    def _qident(self, x: str) -> str:
        """Extra helper specifically for dump_table_to_df"""
        return f'"{x.replace(chr(34), chr(34)*2)}"'
    

    def _maybe_cast_int(self, s: pd.Series) -> pd.Series:
        """Strips strings, casts to int, and NaN to None"""
        try:
            if s.dropna().map(lambda v: str(v).strip().lstrip('+').isdigit()).all():
                return s.map(lambda v: None if v is None else int(str(v).strip()))
        except Exception:
            pass
        return s.map(lambda v: v.strip() if isinstance(v, str) else v)


    # ---- Methods for simple/common operations and utils ---- #

    def get_table_columns(self, table_name: str) -> list[str]:
        """Return column names (unquoted) for the given table. Accepts 'schema.table' where applicable; falls back to current schema/db."""

        # Ensure connection
        self._ensure_cxn()

        # Get the schema and table name from the given table name
        schema, tbl = self._split_qualified(table_name)
        
        try:
            # Init cursor
            cursor:DBCursor = self.cxn.cursor()

            # Act based on DB type
            match self.database_type:

                # MYSQL 
                case DatabaseType.MYSQL:

                    # Use current DB if schema not provided
                    if schema:
                        cursor.execute("""
                            SELECT COLUMN_NAME
                            FROM information_schema.columns
                            WHERE TABLE_SCHEMA = %s
                            AND TABLE_NAME   = %s
                            ORDER BY ORDINAL_POSITION
                        """, (schema, tbl))
                    else:
                        cursor.execute("""
                            SELECT COLUMN_NAME
                            FROM information_schema.columns
                            WHERE TABLE_SCHEMA = DATABASE()
                            AND TABLE_NAME   = %s
                            ORDER BY ORDINAL_POSITION
                        """, (tbl,))
                    rows = cursor.fetchall()
                    return [r[0] for r in rows]

                # POSTGRESQL
                case DatabaseType.POSTGRESQL:

                    # Prefer information_schema to handle case/visibility cleanly
                    if schema:
                        cursor.execute("""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = %s
                            AND table_name   = %s
                            ORDER BY ordinal_position
                        """, (schema, tbl))
                    else:
                        cursor.execute("""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = current_schema()
                            AND table_name   = %s
                            ORDER BY ordinal_position
                        """, (tbl,))
                    rows = cursor.fetchall()
                    return [r[0] for r in rows]

                # SQLITE
                case DatabaseType.SQLITE:

                    # PRAGMA needs the identifier in the SQL, so use the quoting helper
                    identifier:str = self._quote_identifier(tbl) if not schema else self._quote_identifier(f"{schema}.{tbl}")
                    cursor.execute(f"PRAGMA table_info({identifier})")
                    rows:list[tuple] = cursor.fetchall()
                    
                    # NOTE: returned format for PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
                    # i.e. name is at index 1
                    return [r[1] for r in rows]

                # UNSUPPORTED
                case _:
                    return []
                
        # Handle exceptions
        except Exception as e:
            self.log_error("get_table_columns", e)
            return []
        
        # When all done 
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass


    def get_all_table_names(self) -> list[str]:
        """Returns all *user* table names (unquoted) visible in the current DB/schema."""

        # Ensure connection
        self._ensure_cxn()

        try:
            # Init cursor
            cursor:DBCursor = self.cxn.cursor()

            # Act based on db type
            match self.database_type:

                # MYSQL
                case DatabaseType.MYSQL:

                    # Current database only; excludes views
                    cursor.execute("""
                        SELECT TABLE_NAME
                        FROM information_schema.tables
                        WHERE TABLE_TYPE = 'BASE TABLE'
                        AND TABLE_SCHEMA = DATABASE()
                        ORDER BY TABLE_NAME
                    """)
                    rows = cursor.fetchall()
                    return [r[0] for r in rows]

                # POSTGRESQL 
                case DatabaseType.POSTGRESQL:

                    # Current schema only; excludes system schemas and views
                    cursor.execute("""
                        SELECT c.relname AS table_name
                        FROM pg_catalog.pg_class c
                        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                        WHERE c.relkind = 'r'        -- ordinary tables
                        AND n.nspname = current_schema()
                        ORDER BY c.relname
                    """)
                    rows = cursor.fetchall()
                    return [r[0] for r in rows]

                # SQLITE
                case DatabaseType.SQLITE:

                    # Exclude SQLite internal tables
                    cursor.execute("""
                        SELECT name
                        FROM sqlite_master
                        WHERE type = 'table'
                        AND name NOT LIKE 'sqlite_%'
                        ORDER BY name
                    """)
                    rows = cursor.fetchall()
                    return [r[0] for r in rows]

                # UNSUPPORTED 
                case _:
                    return []
                
        # Handle exceptions
        except Exception as e:
            self.log_error("get_all_table_names", e)
            return []
        
        # When all done
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass


    def table_as_df(self, table_name:str) -> pd.DataFrame|None: 
        """Returns the given table from the DB as a DataFrame."""
        
        # Make sure DB is connected 
        self._ensure_cxn()
        cursor:DBCursor = self.cxn.cursor()

        # Clean the table name identifier
        raw_name:str = table_name
        identifier:str = self._quote_identifier(table_name)

        # Make sure table exists
        # NOTE: normalize identifiers to unqualified names by comparing only the last part (e.g. if table_name = "public.users", then table_name -> "users")
        try:
            known:set = set(self.get_all_table_names())
            last:str = raw_name.split(".")[-1]
            
            # Check if the raw name AND last part of the raw name are in the known set of table names 
            if raw_name not in known and last not in known:
                self.log_error("table_as_df", TableDoesNotExist(raw_name))
                return None
            
        # Handle exceptions
        except Exception as e:
            self.log_error("table_as_df", e)
            return None

        # Execute query
        try: 
            cursor.execute(f'SELECT * FROM {identifier}')
            
            # Convert results into a df and return
            return pd.DataFrame(
                cursor.fetchall(),
                columns=self.get_table_columns(raw_name)    # NOTE: get cols using the raw name 
            )

        # Handle exceptions
        except Exception as e: 
            self.log_error('table_as_df()', e)
            return None

        # When all done
        finally: 
            if cursor is not None: 
                try: cursor.close()
                except Exception:
                    pass    # Make sure results are returned if error here 


    def db_as_csvs(self, output_dir:str) -> None: 
        """Downloads all the tables in the DB to CSV files and outputs to the given [output_dir]."""
        
        # Create output dir if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Iterate over each of the table names
        for table in self.get_all_table_names(): 

            # Get the table as a df and save to a CSV in the output dir
            try: 
                self.table_as_df(table).to_csv(os.path.join(output_dir, f'{table}.csv'), index=False)
            except Exception as e: 
                self.log_error('db_as_csvs()', e)
        

    def table_from_df_schema(self, table_name:str, df:pd.DataFrame) -> None: 
        """Creates a new table with the given [table_name] using the schema (including col names and dtypes) from the given [df].
        NOTE: the created table is EMPTY - it just mimics the schema of the df."""
        
        # Ensure connection
        self._ensure_cxn()

        # Base cases: db is empty or empty columns
        if df is None or df.columns.empty:
            self.log_error("table_from_df_schema", ValueError("DataFrame has no columns"))
            return

        # Get the raw table name and validate it
        raw_name:str = (table_name or "").strip()
        
        if not raw_name:
            self.log_error("table_from_df_schema", ValueError("Empty table name"))
            return

        # Build column definitions
        col_defs:list[str] = []
        for col in df.columns:
            col_name = str(col)
            sql_type = self._sql_type_for_dtype(df.dtypes[col_name])
            col_ident = self._quote_identifier(col_name)
            col_defs.append(f"{col_ident} {sql_type}")

        # Assemble CREATE TABLE statement
        table_identifier:str = self._quote_identifier(raw_name)
        create_sql = f"CREATE TABLE IF NOT EXISTS {table_identifier} (\n  " + ",\n  ".join(col_defs) + "\n)"

        # Init a cursor and execute the statement    
        cursor:DBCursor = None
        try:
            cursor = self.cxn.cursor()
            cursor.execute(create_sql)

        # Handle exceptions
        except Exception as e:
            self.log_error("table_from_df_schema", e)
            
            # Roll back on error when supported
            try: self.cxn.rollback()
            except Exception: pass
            return

        # When all done
        finally:
            
            # Commit changes
            self._commit()

            # Close cursor
            try: cursor.close()
            except: pass


    def new_table_row(self, row:Iterable[Iterable[int|str]], table_name:str, cols:list[str]|None=None) -> None: 
        """Creates a new row in the given table with the given values. 
        
            - If an unordered list, tuple, or set is given as [row], then the order MUST match the expected column order
            - If a dictionary is given as [row], then the keys are matched to the column names and the keys MUST match the 
              expected column names
            - If cols are given, that subset of columns is used; otherwise the entire set of the table's columns are used
              (or the keys of the row dictionary)
        """
        
        # Ensure connection
        self._ensure_cxn()
        cursor:DBCursor = self.cxn.cursor()

        # Get the table columns
        table_cols:list[str] = self.get_table_columns(table_name)

        # Check if given cols or if row is a dict
        cols_to_insert:list[str] = None
        
        # Inserting the row keys
        if isinstance(row, dict): 
            cols_to_insert = [self._quote_identifier(c) for c in list(row.keys())]
        # Inserting the given cols
        elif cols is not None: 
            cols_to_insert = [self._quote_identifier(c) for c in cols]
        # Inserting all cols
        else: 
            cols_to_insert = [self._quote_identifier(c) for c in table_cols]

        # Generate query SQL
        query:str = self._generate_insert_statement(table_name, cols=cols_to_insert)

        # Execute query (NOTE: raise_on_error defaults to False)
        self.execute_one(query, row, cursor, close_cursor=True)


    def dump_df_to_table(self, df:pd.DataFrame, table_name:str) -> None: 
        """Dumps all rows in the given df into the given table."""

        # Do nothing if given df is empty 
        if df.empty: 
            self.log_warning('dump_df_to_table()', f'given an empty DataFrame - no new rows created in "{table_name}".')
            return 
        
        # Ensure connection
        self._ensure_cxn()
        cursor:DBCursor = self.cxn.cursor() 
        

        # ---- Validating given params ---- #

        # Make sure the number of columns in the df matches the table to avoid errors later
        db_table_cols:list[str] = self.get_table_columns(table_name)

        if len(df.columns) != len(db_table_cols): 
            self.log_error(f'dump_df_to_table()', IndexError(f'Given df contains {len(df.columns)}, but given table "{db_table_cols}" contains {len(db_table_cols)} columns - skipping execution.'))
            return 
        
        # Reorder/select df columns to exactly match the table
        # (in case df has the same columns but different order)
        try:
            df_ordered:pd.DataFrame = df[db_table_cols]
        except KeyError:
            self.log_error('dump_df_to_table()', IndexError(f'DF columns do not match table columns (DF columns={list(df.columns)}, table columns={db_table_cols})'))
            return
        
        # Convert NaN -> None so DB gets "None" instead of pd.na, and normalize numpy types
        df_ordered = df_ordered.apply(self._maybe_cast_int, axis=0)

        # Generate the INSERT query
        placeholders:str = ",".join([self.P] * len(db_table_cols))
        cols_sql = ",".join(self._qident(c) for c in db_table_cols)
        query = f'INSERT INTO {self._qident(table_name)} ({cols_sql}) VALUES ({placeholders})'
        
        # Build value tuples
        val_tuples = [
            tuple((None if pd.isna(v) else v) for v in row)
            for row in df_ordered.itertuples(index=False, name=None)
        ]

        # Execute query and commit changes
        try: 
            cursor.executemany(query, val_tuples)
            self.log_debug('dump_df_to_table()', f'Inserted {cursor.rowcount} rows into "{table_name}".')
        
        # Handle exceptions
        except Exception as e:

            # Default back to a row-by-row insert, so we insert the rows that work and can identify those that failed
            self.cxn.rollback()
            self.log_error('dump_df_to_table()', e)
            self.log_debug('dump_df_to_table()', 'Falling back to row-by-row insert...')

            for i, tup in enumerate(val_tuples, start=1):
                try:
                    cursor.execute(query, tup)
                except Exception as e2:
                    self.cxn.rollback()
                    self.log_error(
                        'dump_df_to_table()',
                        RuntimeError(
                            f'Row {i} failed for table "{table_name}". Values={tup}. Error={e2}.'
                        )
                    )
                    raise
        
        # When all done
        finally: 

            # Commit changes
            try: 
                self._commit()
                cursor.close() 
            except Exception as e: 
                pass


    # ---- Standard functions ---- #
    def execute_one(
        self, 
        statement:str, 
        params:Iterable[str|int]=None, 
        cursor:SQLiteCursor|MySQLCursor|PSQLCursor=None,
        *, 
        commit:bool=True, 
        close_cursor:bool=True, 
        fetch_results:bool=False,
        rollback_on_error:bool=True,
        raise_on_error:bool=False
    ) -> list[tuple]|None:
        """Executes the given statement using the given params, and logs any errors.
         
            NOTE:
                - If a cursor is given, then that cursor is used; is NOT given, then a cursor is created with [self.cxn]
                - If [close_cursor] is True OR if a cursor is created (i.e. passed [cursor] is None), then the cursor is closed after the transaction
                - If [fetch_results] is True, then the results are fetched and returned as a list of tuples; otherwise, None is returned
        """

        # Check if the cxn is active
        self._ensure_cxn()
        
        # Init vars
        results: Optional[list[tuple]] = None   # Holds results if configured
        created_cursor:bool = False             # Flag that indiciates we created the cursor (not passed)

        # If not given a cursor, create one
        if cursor is None or not cursor: 
            created_cursor = True           # Set flag
            cursor = self.cxn.cursor()      # Create cursor

        # Execute statement
        try: 

            # Execute with parameters
            if params is not None and params: 
                cursor.execute(statement, params)
            
            # Execute without parameters
            else: 
                cursor.execute(statement)

            # Fetch results if configured 
            if fetch_results: 
                results = cursor.fetchall()
            
        # Handle and log errors
        except Exception as e: 
            if rollback_on_error:
                try:
                    self.cxn.rollback()
                except Exception:
                    # NOTE: don't mask the original exception
                    pass

            # Log with traceback
            self.log_error('execute_one()', e)

            # Re-raise if configured
            if raise_on_error: raise e
        
        # Once transaction is done
        finally:

            # Commit if configured
            if commit: self._commit()
            
            # Close cursor if it was created or if configured
            if created_cursor or close_cursor: 
                try: 
                    cursor.close()
                except Exception as e: 
                    self.log_error('execute_one()', e)
        
        # Return the results (defaults to None if fetch_results is False)
        return results
    

    def execute_many(
        self, 
        statement:str, 
        params:Iterable[str|int], 
        cursor:SQLiteCursor|MySQLCursor|PSQLCursor=None,
        *, 
        commit:bool=True, 
        close_cursor:bool=True, 
        fetch_results:bool=False,
        rollback_on_error:bool=True,
        raise_on_error:bool=False
    ) -> list[tuple]|None:
        """Executes the given statement for all given params (DBCursor.executemany()) and logs any errors.
         
            NOTE:
                - If a cursor is given, then that cursor is used; is NOT given, then a cursor is created with [self.cxn]
                - If [close_cursor] is True OR if a cursor is created (i.e. passed [cursor] is None), then the cursor is closed after the transaction
                - If [fetch_results] is True, then the results are fetched and returned as a list of tuples; otherwise, None is returned
        """

        # Check if the cxn is active
        self._ensure_cxn()
        
        # Init vars
        results: Optional[list[tuple]] = None   # Holds results if configured
        created_cursor:bool = False             # Flag that indiciates we created the cursor (not passed)

        # If not given a cursor, create one
        if cursor is None or not cursor: 
            created_cursor = True           # Set flag
            cursor = self.cxn.cursor()      # Create cursor

        # Execute statement
        try: 

            # Execute with parameters
            cursor.executemany(statement, params)

            # Fetch results if configured 
            if fetch_results: 
                results = cursor.fetchall()
            
        # Handle and log errors
        except Exception as e: 
            if rollback_on_error:
                try:
                    self.cxn.rollback()
                except Exception:
                    # NOTE: don't mask the original exception
                    pass

            # Log with traceback
            self.log_error('execute_many()', e)

            # Re-raise if configured
            if raise_on_error: raise e
        
        # Once transaction is done
        finally:

            # Commit if configured
            if commit: self._commit()
            
            # Close cursor if it was created or if configured
            if created_cursor or close_cursor: 
                try: 
                    cursor.close()
                except Exception as e: 
                    self.log_error('execute_many()', e)
        
        # Return the results (defaults to None if fetch_results is False)
        return results