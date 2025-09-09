"""
Kiwicalc - A comprehensive mathematical library for Python

This package provides a friendly API for mathematical computations including:
- Algebraic expressions and operations
- Function plotting and visualization
- Equation solving
- Linear algebra operations
- Probability and statistics
- And much more!

Usage:
    import kiwicalc as kw
    # or
    from kiwicalc import Mono, Poly, Function, Range, plot_function
"""

# Version information
__version__ = "1.0.0"
__author__ = "Kiwicalc Team"

# =============================================================================
# CORE ALGEBRA CLASSES
# =============================================================================

# Base expression interface
from .algebra.IExpression import IExpression

# Basic algebraic classes
from .algebra.mono import Mono
from .algebra.poly import Poly
from .algebra.var import Var
from .algebra.fraction import Fraction
from .algebra.root import Root, Sqrt
from .algebra.exponent import Exponent
from .algebra.abs import Abs
from .algebra.factorial import Factorial
from .algebra.fastpoly import FastPoly

# Complex expressions
from .algebra.expression_sum import ExpressionSum
from .algebra.expression_mul import ExpressionMul
from .algebra.poly_fraction import PolyFraction

# Trigonometric expressions
from .algebra.trigonometry.trigoexpr import TrigoExpr
from .algebra.trigonometry.trigoexprs import TrigoExprs

# Logarithmic expressions
from .algebra.log.log import Log, Ln

# =============================================================================
# FUNCTIONS
# =============================================================================

from .functions.function import Function
from .functions.function_collection import FunctionCollection
from .functions.function_chain import FunctionChain

# =============================================================================
# OPERATORS AND RANGES
# =============================================================================

from .operators.operator import (
    GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL,
    Operator, GreaterThan, LessThan, GreaterOrEqual, LessOrEqual
)

from .range import (
    Range, RangeCollection, RangeOR, RangeAND, 
    create_range, range_operator_from_string
)

# =============================================================================
# PLOTTING AND VISUALIZATION
# =============================================================================

# Main plotting functions
from .plotting.plot import (
    plot_function, plot_functions, plot_function_3d,
    scatter_function, scatter_function_3d, scatter_dots, scatter_dots_3d,
    create_grid, draw_axis, process_to_points
)

# Plotting models
from .plotting.models import IPlottable, IScatterable
from .plotting.models.point import Point, Point1D, Point2D, Point3D, Point4D
from .plotting.models.point_collection import (
    PointCollection, Point1DCollection, Point2DCollection, 
    Point3DCollection, Point4DCollection
)
from .plotting.models.line import Line2D
from .plotting.models.circle import Circle

# =============================================================================
# EQUATIONS AND SOLVING
# =============================================================================

from .equations.Equation import Equation
from .equations.linear_equation import LinearEquation
from .equations.quadratic_equation import QuadraticEquation
from .equations.cubic_equation import CubicEquation
from .equations.quartic_equation import QuarticEquation
from .equations.poly_equation import PolyEquation
from .equations.linear_system import LinearSystem

# Equation utilities
from .equations.auxiliary import (
    equation_to_function, get_equation_variables, extract_dict_from_equation,
    format_coefficient, format_free_number, gcd, float_gcd, solve_poly_system
)

# =============================================================================
# LINEAR ALGEBRA
# =============================================================================

from .linear_algebra.matrices.matrix import Matrix
from .linear_algebra.vectors.vector import Vector
from .linear_algebra.vectors.vector2d import Vector2D
from .linear_algebra.vectors.vector3d import Vector3D
from .linear_algebra.vectors.vector_collection import VectorCollection

# Linear algebra utilities
from .linear_algebra.auxiliary import (
    generate_jacobian, approximate_jacobian, generate_polynomial_matrix
)

# =============================================================================
# PROBABILITY AND STATISTICS
# =============================================================================

from .probability.occurrence import Occurrence
from .probability.probability_tree import ProbabilityTree

# =============================================================================
# SEQUENCES
# =============================================================================

from .sequences.sequence import Sequence
from .sequences.arithmetric_prog import ArithmeticProgression
from .sequences.geometric_seq import GeometricSequence
from .sequences.recursive_seq import RecursiveSequence

# =============================================================================
# MACHINE LEARNING
# =============================================================================

from .machine_learning.linear_regression import linear_regression
from .machine_learning.loss_functions import (
    mean_squared_error, mean_absolute_error, 
    root_mean_squared_error, mean_absolute_percentage_error
)

# =============================================================================
# NUMERICAL METHODS
# =============================================================================

from .numerical.numerical import (
    newton_raphson, aberth_method, lagrange_polynomial, 
    taylor_polynomial, numerical_diff, monic_poly_from_coefficients
)

# =============================================================================
# CALCULUS
# =============================================================================

from .calculus.calculus import derivative, integral
from .calculus.integrals import definite_integral, indefinite_integral

# =============================================================================
# WORKSHEETS AND EXERCISES
# =============================================================================

from .worksheets.exercises import (
    generate_linear_equations, generate_quadratic_equations,
    generate_polynomial_equations, generate_system_of_equations
)

from .worksheets.pdfworksheet import PDFWorksheet, worksheet
from .worksheets.pdfexercise import PDFExercise
from .worksheets.pdfpage import PDFPage

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

from .auxiliary import (
    round_decimal, decimal_range, values_in_range, clean_spaces,
    contains_from_list, extract_variables_from_expression, copy_expression,
    only_numbers_letters, create, create_from_dict, get_factors,
    synthetic_division, equal_ignore_order
)

from .global_functions import factorial

from .string_analysis import (
    handle_parenthesis, handle_abs, handle_factorial, is_evaluatable,
    is_number, split_expression, formatted_expression, to_lambda,
    apply_parenthesis, extract_coefficient, _format_minus
)

# =============================================================================
# CONSTANTS
# =============================================================================

from .constants import (
    MATHEMATICAL_CONSTANTS, TRIGONOMETRY_CONSTANTS,
    E, PI, TAU, INFINITY, NEGATIVE_INFINITY
)

# =============================================================================
# CONVENIENT ALIASES
# =============================================================================

# Short aliases for common classes
M = Mono
P = Poly
V = Var
F = Function
R = Range
RC = RangeCollection
ES = ExpressionSum
EM = ExpressionMul

# Mathematical constants
e = E
pi = PI
tau = TAU

# =============================================================================
# PUBLIC API - What gets imported with "from kiwicalc import *"
# =============================================================================

__all__ = [
    # Core algebra
    "IExpression", "Mono", "Poly", "Var", "Fraction", "Root", "Sqrt", 
    "Exponent", "Abs", "Factorial", "FastPoly", "ExpressionSum", "ExpressionMul",
    "PolyFraction", "TrigoExpr", "TrigoExprs", "Log", "Ln",
    
    # Functions
    "Function", "FunctionCollection", "FunctionChain",
    
    # Operators and ranges
    "GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL",
    "Operator", "GreaterThan", "LessThan", "GreaterOrEqual", "LessOrEqual",
    "Range", "RangeCollection", "RangeOR", "RangeAND", "create_range",
    
    # Plotting
    "plot_function", "plot_functions", "plot_function_3d", "scatter_function",
    "scatter_function_3d", "create_grid", "draw_axis", "IPlottable", "IScatterable",
    "Point", "Point1D", "Point2D", "Point3D", "Point4D", "PointCollection",
    "Line2D", "Circle",
    
    # Equations
    "Equation", "LinearEquation", "QuadraticEquation", "CubicEquation",
    "QuarticEquation", "PolyEquation", "LinearSystem",
    
    # Linear algebra
    "Matrix", "Vector", "Vector2D", "Vector3D", "VectorCollection",
    
    # Probability
    "Occurrence", "ProbabilityTree",
    
    # Sequences
    "Sequence", "ArithmeticProgression", "GeometricSequence", "RecursiveSequence",
    
    # Machine learning
    "linear_regression", "mean_squared_error", "mean_absolute_error",
    
    # Numerical methods
    "newton_raphson", "aberth_method", "lagrange_polynomial", "taylor_polynomial",
    
    # Calculus
    "derivative", "integral", "definite_integral", "indefinite_integral",
    
    # Utilities
    "round_decimal", "decimal_range", "values_in_range", "clean_spaces",
    "create", "to_lambda", "is_number", "formatted_expression",
    "factorial", "float_gcd", "gcd",
    
    # Constants
    "E", "PI", "TAU", "INFINITY", "NEGATIVE_INFINITY",
    
    # Short aliases
    "M", "P", "V", "F", "R", "RC", "ES", "EM", "e", "pi", "tau",
]

# =============================================================================
# PACKAGE METADATA
# =============================================================================

__title__ = "kiwicalc"
__description__ = "A comprehensive mathematical library for Python"
__url__ = "https://github.com/kiwicalc/kiwicalc"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Kiwicalc Team"
