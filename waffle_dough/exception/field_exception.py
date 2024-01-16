from .base_exception import BaseException


class FieldException(BaseException):
    """Base class for exceptions in `Field` module."""


class FieldValidationError(FieldException):
    """Exception raised for errors in the field validation."""


class FieldTaskError(FieldException):
    """Exception raised for errors in the task validation."""


class FieldMissingError(FieldException):
    """Exception raised for errors in the missing fields."""
