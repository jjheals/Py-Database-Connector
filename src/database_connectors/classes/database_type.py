from enum import Enum


class DatabaseType(Enum): 
    """Enum of Database types for standardization and type checking."""
    MYSQL = 1
    POSTGRESQL = 2
    SQLITE = 3
