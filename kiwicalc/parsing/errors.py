class EquationParseError(ValueError):
    """Base error for invalid equation or expression syntax."""


class UnsupportedExpressionError(EquationParseError):
    """Raised when valid-looking syntax cannot be represented by an API."""


class AmbiguousVariableError(EquationParseError):
    """Raised when a variable name cannot be resolved unambiguously."""
