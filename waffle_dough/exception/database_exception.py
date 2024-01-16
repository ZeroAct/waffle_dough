from .base_exception import BaseException


class DatabaseException(BaseException):
    """Base class for exceptions in `Database` module."""


class DatabaseConnectionError(DatabaseException):
    """Exception raised for errors in the database connection."""


class DatabaseQueryError(DatabaseException):
    """Exception raised for errors in the database query."""


class DatabaseNotFoundError(DatabaseException):
    """Exception raised for errors in the database not found."""


class DatabaseDuplicateError(DatabaseException):
    """Exception raised for errors in the database duplicate."""


class DatabaseIntegrityError(DatabaseException):
    """Exception raised for errors in the database integrity."""


class DatabaseConstraintError(DatabaseException):
    """Exception raised for errors in the database constraint."""
