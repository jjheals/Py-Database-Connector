from __future__ import annotations
from typing import Protocol, Any, Mapping, Sequence, Iterator, runtime_checkable

# NOTE: rows can be tuples (default) or dict-like 
Row = Any  

@runtime_checkable
class DBCursor(Protocol):

    # Common DB-API attributes
    description: Any | None
    rowcount: int

    # Core execution methods
    def execute(
        self,
        operation: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> Any: ...

    def executemany(
        self,
        operation: str,
        seq_of_params: Sequence[Sequence[Any] | Mapping[str, Any]],
    ) -> Any: ...

    # Fetch methods
    def fetchone(self) -> Row | None: ...
    def fetchmany(self, size: int = ...) -> list[Row]: ...
    def fetchall(self) -> list[Row]: ...

    # Lifecycle
    def close(self) -> None: ...

    # Iteration & context manager support 
    def __iter__(self) -> Iterator[Row]: ...
    def __enter__(self) -> "CursorLike": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...

