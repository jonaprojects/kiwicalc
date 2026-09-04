from kiwicalc.core import *
from kiwicalc.parsing import *
from kiwicalc.expressions import *
from kiwicalc.numeric import *
from kiwicalc.geometry import *
from kiwicalc.linalg import *
from kiwicalc.equations import *
from kiwicalc.functions import *
from kiwicalc.sequences import *
from kiwicalc.probability import *
from kiwicalc.plotting import *
from kiwicalc.pdf import *

__all__ = [
    # Core
    'TRIGONOMETRY_CONSTANTS', 'MATHEMATICAL_CONSTANTS', 'IExpression',
    'IPlottable', 'IScatterable', 'IPlottable3D', 'IScatterable3D',
    'Range', 'RangeCollection', 'RangeOR', 'RangeAND', 'create_range',
    'factorial', 'float_gcd', 'gamma', 'round_decimal', 'to_lambda', 'decimal_range',
    # Expressions
    'Var', 'Mono', 'Poly', 'FastPoly', 'ExpressionSum', 'ExpressionMul',
    'Fraction', 'PolyFraction', 'Root', 'Sqrt', 'Log', 'PolyLog', 'Ln',
    'TrigoExpr', 'TrigoExprs', 'Sin', 'Cos', 'Tan', 'Cot', 'Sec', 'Csc',
    'Asin', 'Acos', 'Atan', 'Acot', 'ASec', 'ACsc',
    'Factorial', 'Abs', 'Exponent', 'create', 'create_from_dict',
    # Equations
    'Equation', 'LinearEquation', 'QuadraticEquation', 'CubicEquation',
    'QuarticEquation', 'PolyEquation', 'LinearSystem',
    'solve_linear', 'solve_quadratic', 'solve_quadratic_real', 'solve_quadratic_params',
    'solve_cubic', 'solve_cubic_real', 'solve_quartic', 'solve_polynomial',
    'solve_poly_by_factoring', 'solve_linear_system', 'solve_poly_system',
    'random_linear', 'random_polynomial', 'random_polynomial2',
    'random_linear_system', 'random_poly_system',
    # Functions
    'Function', 'FunctionCollection', 'FunctionChain',
    # Geometry
    'Point', 'Point1D', 'Point2D', 'Point3D', 'Point4D', 'Line2D', 'Circle',
    'PointCollection', 'Point1DCollection', 'Point2DCollection', 'Point3DCollection', 'Point4DCollection',
    'Vector', 'Vector2D', 'Vector3D', 'VectorCollection',
    'Curve2D', 'TransformedCurve2D', 'ParametricCurve2D', 'PolarCurve2D', 'ImplicitCurve2D',
    'BezierCurve2D', 'CatmullRomSpline2D', 'Ellipse', 'Arc', 'Parabola',
    'Hyperbola', 'ArchimedeanSpiral', 'LogarithmicSpiral', 'LissajousCurve2D',
    'Cardioid', 'RoseCurve', 'Cycloid', 'Epicycloid', 'Hypocycloid',
    'Superellipse', 'Catenary', 'Involute',
    'Curve3D', 'TransformedCurve3D', 'ParametricCurve3D', 'BezierCurve3D', 'CatmullRomSpline3D',
    'Line3D', 'Helix', 'LissajousCurve3D', 'TorusKnot', 'TrefoilKnot', 'FigureEightKnot',
    'Surface3D', 'ExplicitSurface3D', 'ParametricSurface3D',
    'Sphere', 'Ellipsoid', 'Cylinder', 'Cone', 'Torus',
    'Paraboloid', 'HyperbolicParaboloid', 'Hyperboloid',
    # Linear Algebra
    'Matrix', 'Surface', 'column', 'mav', 'msv', 'mrv',
    'LinearSolveResult', 'LUDecomposition', 'QRDecomposition', 'SVDDecomposition',
    'EigenDecomposition', 'VectorSpaceBasis', 'GramSchmidtStep', 'GramSchmidtResult',
    'ProjectionResult', 'RowOperation', 'RowReductionExplanation', 'LinearAlgebraPlot',
    'AffineTransformation',
    # Sequences
    'Sequence', 'GeometricSeq', 'ArithmeticProg', 'RecursiveSeq',
    # Probability
    'Occurrence', 'ProbabilityTree',
    # Numeric & Calculus
    'newton_raphson', 'halleys_method', 'secant_method', 'inverse_interpolation',
    'laguerre_method', 'durand_kerner', 'durand_kerner2', 'ostrowski_method',
    'chebychevs_method', 'aberth_method', 'steffensen_method', 'bisection_method',
    'bairstow_method', 'reinman', 'trapz', 'simpson', 'numerical_diff',
    'gradient_descent', 'gradient_ascent',
    'NumericalResult', 'differentiate', 'integrate', 'find_root',
    'gradient', 'jacobian', 'hessian', 'solve_system', 'integrate_nd',
    'differentiate_samples', 'cumulative_integrate',
    'solve_ivp', 'ODESolution', 'ODEEvent',
    'NumericalStep', 'NumericalExplanation', 'NumericalAnimation',
    # Plotting
    'plot_function', 'plot_functions', 'plot_function_3d', 'plot_functions_3d',
    'scatter_function', 'scatter_functions', 'scatter_function_3d', 'scatter_functions_3d',
    'scatter_dots', 'scatter_dots_3d',
    'plot_vector_2d', 'plot_vector_3d', 'plot_complex', 'plot_multiple',
    'plot_curve_2d', 'scatter_curve_2d', 'plot_implicit_curve_2d',
    'plot_curve_3d', 'scatter_curve_3d', 'plot_surface_3d',
    'Graph', 'Graph2D', 'Graph3D',
    'GraphAnimation', 'GraphInteraction',
    'PlotTheme', 'THEMES', 'available_themes', 'get_theme',
    # PDF
    'PDFWorksheet', 'PDFPage', 'PDFExercise', 'worksheet', 'create_pdf', 'create_pages',
    'PDFMath', 'PDFPlot', 'PDFText', 'PDFArray', 'PDFMatrix', 'PDFVector',
    'PDFStyle', 'PDFHeading', 'PDFAnswerSpace', 'PDFDocument', 'PDFFooter',
    'PDFTheme', 'PDFThemeColors', 'PDFThemeTypography', 'PDFThemeSpacing', 'PDF_THEMES',
    'available_pdf_themes', 'get_pdf_theme', 'format_math', 'format_polynomial',
    'PDFLinearIntersection', 'PDFTrigonometricEquation', 'PDFLogarithmicEquation',
    'PDFAlgebraExercise', 'PDFSimplifyExpression', 'PDFExpandExpression',
    'PDFFactorPolynomial', 'PDFCompleteSquare', 'PDFSubstitution',
    'PDFLinearInequality', 'PDFAbsoluteValueEquation', 'PDFExponentLaws',
    'PDFRationalEquation', 'PDFRadicalEquation', 'PDFRearrangeFormula',
    'ALGEBRA_EXERCISE_TYPES', 'algebra_exercise',
    'PDFCalculusNumericalExercise', 'PDFDifferenceQuotient', 'PDFDerivativeExercise',
    'PDFTangentLine', 'PDFCriticalPoints', 'PDFMonotonicity', 'PDFConcavity',
    'PDFOptimization', 'PDFDefiniteIntegral', 'PDFAreaBetweenCurves',
    'PDFNumericalDerivative', 'PDFTrapezoidalRule', 'PDFSimpsonRule',
    'PDFNewtonIteration', 'PDFEulerMethod', 'PDFRungeKuttaMethod',
    'CALCULUS_EXERCISE_TYPES', 'calculus_exercise',
    'PDFLinearAlgebraExercise', 'PDFVectorArithmetic', 'PDFDotProduct',
    'PDFVectorMagnitude', 'PDFUnitVector', 'PDFMatrixArithmetic',
    'PDFScalarMatrixMultiplication', 'PDFMatrixMultiplicationExercise',
    'PDFDeterminantExercise', 'PDFInverseMatrix', 'PDFLinearSystemExercise',
    'PDFRowReduction', 'PDFMatrixRank', 'PDFLinearIndependence',
    'PDFBasisCoordinates', 'PDFEigenvaluesExercise', 'PDFEigenvectorExercise',
    'PDFVectorProjection', 'PDFLinearTransformationExercise',
    'LINEAR_ALGEBRA_EXERCISE_TYPES', 'linear_algebra_exercise',
    'PDFGeometryExercise', 'PDFDistanceBetweenPoints', 'PDFMidpoint', 'PDFSlope',
    'PDFLineEquation', 'PDFPointLineDistance', 'PDFParallelPerpendicularLines',
    'PDFTriangleArea', 'PDFTriangleCentroid', 'PDFPythagoreanTheorem',
    'PDFCircleEquation', 'PDFArcAndSector', 'PDFPolygonAngles',
    'PDFSolidMeasurement', 'PDFCoordinateTransformation', 'PDFVectorFromPoints',
    'PDFVectorRelationship', 'PDFVectorAngle', 'PDFCrossProduct', 'PDFVectorLine',
    'PDFPlaneEquation', 'GEOMETRY_EXERCISE_TYPES', 'geometry_exercise',
    'PDFSequenceSeriesExercise', 'PDFIdentifySequence', 'PDFArithmeticNextTerms',
    'PDFArithmeticNthTerm', 'PDFArithmeticDifference', 'PDFArithmeticSum',
    'PDFArithmeticMissingTerm', 'PDFGeometricNextTerms', 'PDFGeometricNthTerm',
    'PDFGeometricRatio', 'PDFGeometricSum', 'PDFInfiniteGeometricSum',
    'PDFRecursiveSequence', 'PDFFibonacciSequence', 'PDFSigmaEvaluation',
    'PDFSequenceLimit', 'PDFConvergenceClassification', 'PDFPSeries',
    'PDFGeometricSeriesTest', 'PDFAlternatingSeries', 'PDFTelescopingSeries',
    'PDFElementaryFunctionLimit', 'PDFEulerLimit', 'PDFRemovableLimit',
    'PDFStandardTrigLimit',
    'SEQUENCE_SERIES_EXERCISE_TYPES', 'sequence_exercise'
]
