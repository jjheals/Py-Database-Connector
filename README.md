# DatabaseConnectors

A lightweight and extensible Python package for connecting to and working with multiple SQL databases (**MySQL, PostgreSQL, SQLite**) in a consistent way.  

The core of the package is the `DatabaseConnector` class, which provides:

- Easy connection management for supported databases  
- Logging of queries, warnings, and errors  
- Integration with **Pandas DataFrames** for reading/writing tables  
- Utilities for creating tables from DataFrame schemas, dumping entire databases to CSV, and safe query execution  

---

## Features

- Unified interface across **MySQL**, **PostgreSQL**, and **SQLite**  
- Automatic safe quoting of identifiers to prevent SQL injection risks  
- Robust connection checking and error handling  
- Conversion between **Pandas DataFrames** and SQL tables:
  - `table_as_df()` → Load a table into a DataFrame  
  - `dump_df_to_table()` → Bulk insert a DataFrame into a table  
  - `table_from_df_schema()` → Create new SQL tables from DataFrame schemas  
- Database export tools (`db_as_csvs()`)  
- Flexible query execution:
  - `execute_one()` for single queries  
  - `execute_many()` for batch operations  

---

## Installation

Clone this repo and install locally with `pip`:

```bash
git clone https://github.com/yourusername/DatabaseConnectors.git
cd DatabaseConnectors
pip install -e .
```

---

## Quick Start

```python

from database_connectors.classes.database_connector import DatabaseConnector
from database_connectors.classes.database_type import DatabaseType
import pandas as pd

# Connect to a PostgreSQL database
db = DatabaseConnector(
    database_type=DatabaseType.POSTGRESQL,
    host="localhost",
    port=5432,
    username="user",
    password="password",
    database="mydb"
)

# Check connection
if db.is_connected():
    print("Connected!")

# List tables
print(db.get_all_table_names())

# Load a table into Pandas
df = db.table_as_df("users")
print(df.head())

# Dump a DataFrame into an existing table
new_data = pd.DataFrame({"id": [4, 5], "name": ["Alice", "Bob"]})
db.dump_df_to_table(new_data, "users")

```

---

## Project Structure

```
DatabaseConnectors/
│   pyproject.toml
│
├── src/
│   └── database_connectors/
│       ├── exceptions.py
│       ├── validators.py
│       ├── __init__.py
│       │
│       ├── classes/
│       │   ├── database_connector.py   # Core DatabaseConnector class
│       │   ├── database_type.py        # Enum for supported DBs
│       │   ├── db_cursor.py            # Cursor abstraction
│       │   └── __init__.py
│       │
│       └── utils/
│           └── general.py              # Logging setup and utilities
│
└── tests/
    └── test_basic.py
```

---

## Testing 

Run the test suite with: 

```bash
pytest tests/
```

--- 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
