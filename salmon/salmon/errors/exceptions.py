"""Module for the classes defining salmon exceptions."""


class ModelCardError(Exception):
    """Raise when there is a model card validation error."""


class CacheError(Exception):
    """Raise when there is a cache error."""


class NotFoundInRepository(Exception):
    """Raise when a file is not found in the repository."""


class SalmonConfigError(Exception):
    """Raise when there is a configuration error."""
