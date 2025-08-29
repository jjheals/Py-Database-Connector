
class DatabaseNotConnected(ConnectionError): 
    """Raised when the AbsDatabaseConnection (or child) attempts a transaction but does not have an active [cxn] attribute."""
    
    def __init__(self): 
        super().__init__('The database is not connected or the connection is not healthy.')


class TableDoesNotExist(LookupError):
    """Raised when a DatabaseConnection tries to access a table that does not exist in the DB."""

    def __init__(self, given_table_name:str):
        self.given_table_name = given_table_name or ""
        pretty_name = f"{self.given_table_name} " if self.given_table_name else ""
        super().__init__(f"The given table {pretty_name}does not exist in the database.")


class DatabaseTypeNotSupported(ValueError): 
    """Raised when a DatabaseConnection object has a database_type that is not one of the Enum values in the DatabaseType class."""

    def __init__(self, db_type:str|int): 
        self.db_type = db_type
        super().__init__(f'The current database_type "{db_type}" is not supported. See the DatabaseType enum class for supported types.')