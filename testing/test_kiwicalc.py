import unittest
import kiwicalc as kw
import numpy as np
import math


# Unit Testing for the project

class TestOutsideMethods(unittest.TestCase):
    # Test for example
    def test_factorial(self):
        self.assertEqual(kw.factorial(3), 6)
        self.assertEqual(kw.factorial(0), 1)
        self.assertEqual(kw.factorial(4), 24)
        self.assertEqual(kw.factorial(1), 1)
        self.assertEqual(kw.factorial(2), 2)

    def test_float_gcd(self):
        self.assertEqual(kw.float_gcd(3.5, 0.5), 0.5)
        self.assertEqual(kw.float_gcd(0, 0.5), 0)
        self.assertEqual(kw.float_gcd(-2, -4), -2)


class TestMono(unittest.TestCase):
    """ check whether the Mono Class operates as it should"""

    def test_add(self):
        my_mono = kw.Mono("3x^2")
        other_mono = kw.Mono("2x^2")
        self.assertEqual(my_mono + other_mono, kw.Mono("5x^2"))
        my_mono += kw.Mono("2x^2")
        self.assertEqual(my_mono, kw.Mono("5x^2"))

        x = kw.Var('x')
        self.assertEqual(my_mono + 5, 5 * x ** 2 + 5)
        self.assertEqual(my_mono + kw.Factorial(5), 5 * x ** 2 + 120)
        self.assertEqual(my_mono + 5, 5 + my_mono)

    def test_sub(self):
        my_mono = kw.Mono("3x^2")
        other_mono = kw.Mono("2x^2")
        self.assertEqual(my_mono - other_mono, kw.Mono("x^2"))
        my_mono -= kw.Mono("4x^2")
        self.assertEqual(my_mono, kw.Mono("-x^2"))

        x = kw.Var('x')
        self.assertEqual(my_mono - 5, -x ** 2 - 5)
        self.assertEqual(my_mono - kw.Factorial(5), - x ** 2 - 120)
        self.assertEqual(3 * x - 3 * x, kw.Mono(0))
        self.assertEqual(3 * x - 0, 3 * x)

    def test_multiply(self):
        my_mono = kw.Mono("5x")
        other_mono = kw.Mono("4x^3")
        self.assertEqual(my_mono * other_mono, kw.Mono("20x^4"))
        x = kw.Var('x')
        self.assertEqual(3 * x ** 2 * 5, kw.Mono("15x^2"))
        self.assertEqual(kw.Mono(5) * kw.Mono(7), kw.Mono(35))
        self.assertEqual(kw.Mono("7x") * kw.Mono("-5x"), kw.Mono("-35x^2"))
        y = kw.Var('y')
        self.assertEqual(2 * y ** 2 * 3 * y, 6 * y ** 3)
        self.assertEqual(kw.Mono("3xy"), 3 * x * y)
        self.assertEqual(kw.Mono("4xy^2") * kw.Mono("5x"), 20 * x ** 2 * y ** 2)

    def test_divide(self):
        x = kw.Var('x')
        y = kw.Var('y')
        my_mono = kw.Mono("30x^2")
        self.assertEqual(my_mono / 5, 6 * x ** 2)
        self.assertEqual(my_mono / x, 30 * x)
        self.assertEqual(my_mono / (30 * x), x)
        self.assertEqual(3 * x * y / 3, x * y)
        self.assertEqual((3 * x ** 2 * y ** 3) / (2 * x * y), 1.5 * x * y ** 2)

    def test_try_evaluate(self):
        x = kw.Var('x')
        my_mono = kw.Mono("4x^3")
        self.assertEqual(my_mono.try_evaluate(), None)
        self.assertEqual(my_mono.when(x=3).try_evaluate(), 108)
        other_mono = kw.Mono(6)
        self.assertEqual(other_mono.try_evaluate(), 6)
        self.assertEqual((3 * x - 3 * x).try_evaluate(), 0)

    def test_equal(self):
        my_mono = kw.Mono("3x")
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(my_mono == 3 * x, True)
        self.assertEqual((3 * x ** 2) / x == 3 * x, True)
        self.assertEqual(x * y * 2 == 2 * x * y, True)
        self.assertEqual(x * 0 == y * 0, True)

    def test_assign(self):
        x = kw.Var('x')
        y = kw.Var('y')
        my_expression = (3 * x ** 2)
        my_expression.assign(x=2)
        self.assertEqual(my_expression, kw.Mono(coefficient=12))
        other_expr = 3 * x * y
        # other_expr.assign(x=y) # Extra feature to add soon
        # self.assertEqual


class TestPoly(unittest.TestCase):
    def test_init(self):
        pass

    def test_add(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual(x + 5, kw.Poly("x+5"))
        self.assertEqual(x ** 2 + 7 + x ** 2 + 1, 2 * x ** 2 + 8)
        self.assertEqual(x ** 2 + 2 * x * y + y ** 2, y ** 2 + 2 * x * y + x ** 2)
        self.assertEqual(x ** 2 + 2 * x * y + y ** 2, kw.Poly("x^2 + 2xy +y^2"))

    def test_sub(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual((x - 5 + y) == (x + y - 5), True)
        self.assertEqual(kw.Poly("5-x^2"), kw.Poly("-x^2+5"))
        self.assertEqual((kw.Poly("x^2-y^2")) == ((x - y) * (x + y)), True)

    def test_mul(self):
        x = kw.Var('x')
        self.assertEqual((x + 2) * (x + 4), x ** 2 + 6 * x + 8)
        self.assertEqual(kw.Poly("x^2 - 4"), kw.Poly("x^2-4"), kw.Poly("x^4-16"))
        self.assertEqual((x ** 2 + 6 * x + 8) * 3, 3 * x ** 2 + 18 * x + 24)

    def test_div(self):
        x = kw.Var('x')
        self.assertEqual((x + 4) * (x + 2) / (x + 4), (x + 2))
        self.assertEqual((x ** 2 + 6 * x + 8) / (x + 4), x + 2)
        self.assertEqual((x ** 2 + 6 * x + 8).__truediv__(x + 4, get_remainder=True), (x + 2, 0))
        self.assertEqual((x ** 2 + 6 * x) / 2, 0.5 * x ** 2 + 3 * x)
        print((x ** 2 + 6 * x + 8) * (x + 5) / (x + 5))
        print((x ** 2 + 6 * x + 8) / (x + 1))
        self.assertEqual((x ** 2 + 6 * x + 8) / (x + 5) * (x + 5), x ** 2 + 6 * x + 8)

    def test_coefficients(self):
        self.assertEqual(kw.Poly("x^3 + 2x^2 + 5").coefficients(), [1, 2, 0, 5])
        self.assertEqual(kw.Poly("2").coefficients(), [2])

    def test_try_evaluate(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual(kw.Poly("3").try_evaluate(), 3)
        other_expression = x ** 2 + 6 - x ** 2
        self.assertEqual(other_expression.try_evaluate(), 6)
        self.assertEqual((2 * x).when(x=5).try_evaluate(), 10)
        self.assertEqual((2 * x ** 2 * y ** 2 + 7).when(y=3).try_evaluate(), None)
        self.assertEqual((2 * x ** 2 * y ** 2 + 7).when(y=3, x=1).try_evaluate(), 25)

    def test_equal(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual(x + y == y + x, True)
        self.assertEqual(x ** 2 + 2 * x * y + y ** 2 == (x + y) ** 2, True)
        self.assertEqual(x == y, False)
        self.assertEqual(x == x, True)
        self.assertEqual(x == x + 5, False)


class TestMatrix(unittest.TestCase):
    def test_init(self):
        self.assertEqual(kw.Matrix("3x3"), kw.Matrix("3, 3"))
        print(kw.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    def test_add(self):
        first_matrix = kw.Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        second_matrix = kw.Matrix([[2, 2, 2], [1, 1, 1], [0, 0, 0]])
        self.assertEqual(first_matrix + second_matrix, kw.Matrix([[3, 3, 3], [3, 3, 3], [3, 3, 3]]))

    def test_sub(self):
        first_matrix = kw.Matrix([[3, 3, 3], [2, 2, 2], [3, 3, 3]])
        second_matrix = kw.Matrix([[2, 2, 2], [1, 1, 1], [0, 0, 0]])
        self.assertEqual(first_matrix - second_matrix, kw.Matrix([[1, 1, 1], [1, 1, 1], [3, 3, 3]]))

    def test_mul(self):
        first_matrix = kw.Matrix([[1, 1], [2, 2]])
        second_matrix = kw.Matrix([[3, 3], [4, 4]])
        print(first_matrix * second_matrix)
        self.assertEqual(first_matrix * second_matrix, kw.Matrix([[3, 3], [8, 8]]))

    def test_copy(self):
        first_matrix = kw.Matrix([[1, 1], [2, 2]])
        self.assertEqual(first_matrix, first_matrix.__copy__())

    def test_matmul(self):
        a = kw.Matrix([[1, 2], [1, 5], [2, 1]])
        b = kw.Matrix([[2, 5], [4, 6]])
        self.assertEqual(a @ b, kw.Matrix([[10, 17], [22, 35], [8, 16]]))

    def test_div(self):
        my_matrix = kw.Matrix([[4, 4], [4, 4]])
        self.assertEqual(my_matrix / 2, kw.Matrix([[2, 2], [2, 2]]))

    def test_rank(self):
        pass

    def test_determinant(self):
        my_matrix = kw.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(my_matrix.determinant(), 0)

    def test_inverse(self):
        my_matrix = kw.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(my_matrix.inverse(), None)

    def test_average(self):
        my_matrix = kw.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(my_matrix.average(), 5)

    def test_sum(self):
        my_matrix = kw.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(my_matrix.sum(), 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)
        x = kw.Var('x')
        other_matrix = kw.Matrix([[x, x, x], [x, x, x], [x, x, x]])
        self.assertEqual(other_matrix.sum(), 9 * x)

    def test_min(self):
        my_matrix = [1, 2, 3, 4, 5]
        matrix_obj = kw.Matrix(my_matrix)
        self.assertEqual(min(matrix_obj), matrix_obj.min())
        other_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertEqual(1, kw.Matrix(other_matrix).min())

    def test_max(self):
        my_matrix = [1, 2, 3, 4, 5]
        matrix_obj = kw.Matrix(my_matrix)
        self.assertEqual(max(matrix_obj), matrix_obj.max())
        other_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertEqual(9, kw.Matrix(other_matrix).max())


class TestPoint(unittest.TestCase):
    def test_init(self):
        self.assertEqual(kw.Point([7, 5]) == kw.Point(np.array([7, 5], dtype=int)), True)

    def test_add(self):
        self.assertEqual(kw.Point([4, 5]) + kw.Point([1, 7]), kw.Point([5, 12]))
        my_point = kw.Point([5, 3])
        my_point += kw.Point([-1, 2])
        self.assertEqual(my_point, kw.Point([4, 5]))
        self.assertEqual(kw.Point([1, 1]) + [2, 2], kw.Point([3, 3]))
        self.assertEqual(kw.Point([1, 1]) + np.array([2, 2]), kw.Point([3, 3]))
        self.assertEqual(kw.Point([1, 1]) + range(1, 3), kw.Point([2, 3]))

    def test_sub(self):
        self.assertEqual(kw.Point([4, 5]) - kw.Point([1, 7]), kw.Point([3, -2]))
        my_point = kw.Point([5, 3])
        my_point -= kw.Point([-1, 2])
        self.assertEqual(my_point, kw.Point([6, 1]))
        self.assertEqual(kw.Point([1, 1]) - [2, 2], kw.Point([-1, -1]))
        self.assertEqual(kw.Point([1, 1]) - np.array([2, 2]), kw.Point([-1, -1]))
        self.assertEqual(kw.Point([1, 1]) - range(1, 3), kw.Point([0, -1]))

    def test_max_coords(self):
        self.assertEqual(kw.Point([1, 3, 2]).max_coord(), 3)

    def test_min_coords(self):
        self.assertEqual(kw.Point([2, 3, 1]).min_coord(), 1)

    def test_sum(self):
        self.assertEqual(kw.Point([6, 8, 5]).sum(), 19)
        x = kw.Var('x')
        self.assertEqual(kw.Point([x, x, x]).sum(), 3 * x)

    def test_eq(self):
        x = kw.Var('x')
        self.assertEqual(kw.Point([6, 4, 2]) == kw.Point([6, 4, 2]), True)
        self.assertEqual(kw.Point([7, 5, 3]) == kw.Point([7, 4, 9]), False)
        self.assertEqual(kw.Point([x + 5, 2 * x - 5, 7 * x + 3]) == kw.Point((x + 5, 2 * x - 5, 7 * x + 3)), True)


class TestPointCollection(unittest.TestCase):
    pass


class TestTrigoExpr(unittest.TestCase):

    def test_init(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual(kw.TrigoExpr("sin(x)*cos(y)"), kw.Sin(x) * kw.Cos(y))
        self.assertEqual(kw.TrigoExpr("3cot(x)"), 3 * kw.Cot(x))
        self.assertEqual(kw.TrigoExpr("2sin(3)*cos(2)"), 2 * kw.Sin(3) * kw.Cos(2))

    def test_add(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.Sin(x) + kw.Sin(x), 2 * kw.Sin(x))
        print(kw.Sin(x) + kw.Cos(y))
        print(kw.TrigoExprs("sin(x) + cos(y)"))
        self.assertEqual(kw.Sin(x) + kw.Cos(y), kw.TrigoExprs("sin(x) + cos(y)"))
        print(kw.Sin(x) + kw.Sin(x))
        print(kw.Sin(x) + kw.Cos(x))
        print(kw.Sin(math.pi / 2) + 4)

    def test_sub(self):
        x, y = kw.Var('x'), kw.Var('y')
        print(kw.Sin(x) - kw.Sin(x))
        self.assertEqual(kw.Sin(x) - kw.Sin(x), 0)
        self.assertEqual(kw.Sin(x) * kw.Cos(y) - kw.TrigoExpr("sin(x)*cos(y)"), 0)
        self.assertEqual(2 * kw.Sin(x) - kw.Sin(x), kw.Sin(x))
        self.assertEqual(3 * kw.Cos(x + 5) - 2 * kw.Sin(x), kw.TrigoExprs("3cos(x+5) - 2sin(x)"))

    def test_mul(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.Sin(2 * x) * kw.Cos(2 * y), kw.TrigoExpr("sin(2x)*cos(2y)"))
        self.assertEqual(kw.Sin(x) * kw.Sin(x), kw.TrigoExpr("sin(x)^2"))
        self.assertEqual(kw.Sin(3) * 5, 5 * kw.Sin(3))
        self.assertEqual(x * kw.Sin(x), kw.Sin(x) * x)

    def test_divide(self):
        x = kw.Var('x')
        self.assertEqual((8 * kw.Sin(x) ** 2 * kw.Cos(x) ** 2) / (2 * kw.Sin(x) * kw.Cos(x)), 4 * kw.Sin(x) * kw.Cos(x))
        self.assertEqual(kw.Sin(x) / kw.Cos(x), kw.Tan(x))
        self.assertEqual(3 * kw.Sin(2 * x) / (kw.Cos(x) * kw.Sin(x)), 6)
        self.assertEqual(4 * kw.Sin(2 * x) * kw.Cos(2 * x) / (kw.Cos(x)), 8 * kw.Sin(x) * kw.Cos(2 * x))
        print(kw.Cos(x) / (kw.Sin(2 * x)), kw.Fraction(1, 2 * kw.Sin(2 * x)))
        print((3 * x * kw.Sin(x)) / kw.Log(x))
        # self.assertEqual((2*kw.Sin(x)**2)/(kw.Cos(x)), 2*kw.Tan(x)*kw.Sin(x))

    def test_power(self):
        pass

    def test_try_evaluate(self):
        self.assertEqual(kw.Sin("3.14").try_evaluate(), kw.sin(3.14))
        self.assertEqual(kw.Sin(3.14).try_evaluate(), kw.sin(3.14))

    def test_equal(self):
        x = kw.Var('x')
        self.assertEqual(kw.Sin(2 * x) == kw.Sin(2 * x), True)
        self.assertEqual(kw.Sin(x) == kw.Sin(x + 2 * math.pi), True)
        self.assertEqual(kw.Sin(x) * kw.Cos(x) == 0.5 * kw.Sin(2 * x), True)
        self.assertEqual(kw.Sin(x) == kw.Sin(x + 5), False)


class TestTrigoExprs(unittest.TestCase):

    def test_init(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.TrigoExprs("sin(x) + cos(y)"), kw.Sin(x) + kw.Cos(y))
        self.assertEqual(kw.Sin(x) + kw.Cos(x), kw.Cos(x) + kw.Sin(x))
        self.assertEqual(kw.Sin(kw.Sin(x)) + 5, 5 + kw.Sin(kw.Sin(x)))

    def test_add(self):
        x, y = kw.Var('x'), kw.Var('y')
        expression = kw.Sin(x) + kw.Cos(x)
        self.assertEqual(expression + 2 * kw.Sin(x), 3 * kw.Sin(x) + kw.Cos(x))
        self.assertEqual(expression + 2 * kw.Sin(x) + 3 * kw.Tan(x), 3 * kw.Sin(x) + kw.Cos(x) + 3 * kw.Tan(x))

    def test_sub(self):
        x, y = kw.Var('x'), kw.Var('y')
        expression = kw.Sin(x) + kw.Cos(x)
        self.assertEqual(expression - 2 * kw.Sin(x), -kw.Sin(x) + kw.Cos(x))
        self.assertEqual(expression - 2 * kw.Sin(x) - 3 * kw.Tan(x), -kw.Sin(x) + kw.Cos(x) - 3 * kw.Tan(x))

    def test_mul(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual((kw.Sin(x) + kw.Cos(y)) * 3, 3 * kw.Sin(x) + 3 * kw.Cos(y))
        self.assertEqual((kw.Sin(x) + kw.Cos(y)) * kw.Sin(x), kw.Sin(x) ** 2 + kw.Cos(y) * kw.Sin(x))
        self.assertEqual((kw.Sin(x) + kw.Cos(y)) * x, x * kw.Sin(x) + x * kw.Cos(y))
        print((kw.Sin(x) + kw.Cos(x)) * (kw.Sin(x) + kw.Cos(x)))
        self.assertEqual((kw.Sin(x) + kw.Cos(x)) * (kw.Sin(x) + kw.Cos(x)),
                         kw.Sin(x) ** 2 + 2 * kw.Cos(x) * kw.Sin(x) + kw.Cos(x) ** 2)

    def test_divide(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual((kw.Sin(x) + kw.Cos(x)) / 2, 0.5 * kw.Sin(x) + 0.5 * kw.Cos(x))
        print((kw.Sin(x) + kw.Cos(x)) / kw.Sin(x))
        # self.assertEqual((kw.Sin(x) + kw.Cos(x))/kw.Sin(x), 1 + kw.Cot(x))

    def test_power(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.Sin(x) * kw.Cos(x) ** 0, kw.Sin(x))

    def test_try_evaluate(self):
        x, y = kw.Var('x'), kw.Var('y')
        expressions = kw.Sin(x) + kw.Cos(y)
        self.assertEqual(expressions.when(x=5, y=4).try_evaluate(), math.sin(5) + math.cos(4))

    def test_simplify(self):
        x, y = kw.Var('x'), kw.Var('y')
        my_expression = kw.TrigoExprs("sin(x)^2 + sin(x)*cos(x) + cos(x)*sin(x)")
        print(my_expression)
        my_expression.simplify()
        print(my_expression)

    def test_assign(self):
        x, y = kw.Var('x'), kw.Var('y')
        print("printing the expression")
        print(((kw.Sin(x) + kw.Cos(y)).when(x=4, y=5)).try_evaluate())
        print((kw.Sin(x) + kw.Cos(y)).when(x=4, y=5))
        self.assertEqual((kw.Sin(x) + kw.Cos(y)).when(x=4, y=5), math.sin(4) + math.cos(5))


class TestExponent(unittest.TestCase):

    def test_try_evaluate(self):
        self.assertEqual(kw.Exponent(2, 2).try_evaluate(), 4)

    def test_add(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(2 * kw.Exponent(3, x) + kw.Exponent(3, x), 3 * kw.Exponent(3, x))
        self.assertEqual(2 ** x + 0, 2 ** x)

    def test_sub(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(4 * kw.Exponent(3, x) - 2 * kw.Exponent(3, x), 2 * kw.Exponent(3, x))
        self.assertEqual(4 * kw.Exponent(3, x) - 4 * kw.Exponent(3, x), 0)
        self.assertEqual(2 ** x - 0, 2 ** x)

    def test_mul(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(3 * kw.Exponent(3, x), kw.Exponent(3, x + 1))
        self.assertEqual(kw.Exponent(3, x) * kw.Exponent(2, x), kw.Exponent(6, x))

    def test_assign(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual((2 ** x).when(x=2), 4)
        self.assertEqual(kw.Exponent(x, y).when(x=5, y=3), 125)


class TestFactorial(unittest.TestCase):
    def test_init(self):
        x = kw.Var('x')
        my_factorial = kw.Factorial(x + 5)

    def test_add(self):
        x = kw.Var('x')
        my_factorial = kw.Factorial(2 * x)
        other_factorial = kw.Factorial(x ** 2 + 6 * x + 8)
        self.assertEqual(isinstance(my_factorial + other_factorial, kw.ExpressionSum), True)
        print(kw.Factorial(5) + kw.Factorial(3))

    def test_sub(self):
        x = kw.Var('x')
        my_factorial = kw.Factorial(2 * x)
        other_factorial = kw.Factorial(x ** 2 + 6 * x + 8)
        self.assertEqual(isinstance(my_factorial - other_factorial, kw.ExpressionSum), True)
        print(kw.Factorial(5) - kw.Factorial(3))

    def test_mul(self):
        x = kw.Var('x')
        self.assertEqual(6 * kw.Factorial(5), kw.Factorial(6))
        print(x * kw.Factorial(x))
        print((x + 1) * kw.Factorial(x))
        self.assertEqual((x + 1) * kw.Factorial(x), kw.Factorial(x + 1))

    def test_div(self):
        x = kw.Var('x')
        self.assertEqual(x * kw.Factorial(x) / 2, 0.5 * x * kw.Factorial(x))
        print(x * kw.Factorial(x) / x)
        self.assertEqual(x * kw.Factorial(x) / x, kw.Factorial(x))
        self.assertEqual(2 * x * kw.Factorial(x) / x, 2 * kw.Factorial(x))

    def test_try_evaluate(self):
        x = kw.Var('x')
        my_factorial = kw.Factorial(6)
        self.assertEqual(my_factorial.try_evaluate(), 720)
        self.assertEqual((kw.Factorial(x).when(x=5)).try_evaluate(), 120)

    def test_eq(self):
        x, y = kw.Var('x'), kw.Var('y')
        pass

    def to_lambda(self):
        pass


class TestLog(unittest.TestCase):
    def test_init(self):
        x = kw.Var('x')
        kw.Log(x)
        kw.Log(kw.Sqrt(x) * kw.Sin(2 * x) + 5)
        kw.Log("log(3x+5,10)", dtype="poly")
        kw.Log("log(sin(x)*cos(x)+ 7, 5)", dtype='trigo')

    def test_try_evaluate(self):
        self.assertEqual(kw.Log(100).try_evaluate(), 2)
        self.assertEqual(kw.Log(1000).try_evaluate(), 3)
        self.assertEqual(kw.Log(10000).try_evaluate(), 4)

    def test_equal(self):
        x = kw.Var('x')
        self.assertEqual(kw.Log(x) == kw.Log(x), True)
        self.assertEqual(kw.Log("log(x)") == kw.Log(x), True)
        self.assertEqual(kw.Log(2) == kw.Log(2), True)
        self.assertEqual(2 * kw.Log(3 * x) == kw.Log(9 * x ** 2), True)
        self.assertEqual(kw.Log(x, math.e), kw.Ln(x), True)

    def test_add(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual(kw.Log(x) + kw.Log(x), kw.Log(x ** 2))
        self.assertEqual(2 * kw.Log(x) + 3 * kw.Log(x), kw.Log(x ** 5))
        self.assertEqual(kw.Log(x) + kw.Log(y), kw.Log(x * y))
        self.assertEqual(kw.Log(10) + kw.Log(100), kw.Mono(3))

    def test_sub(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual(kw.Log(x) - kw.Log(y), kw.Log(x / y))
        self.assertEqual(kw.Log(2 * x) - kw.Log(x), kw.Log(2))
        self.assertEqual(kw.Log(10000) - kw.Log(100), kw.Log(100))

    def test_mul(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual(2 * kw.Log(x), kw.Log(x ** 2))
        self.assertEqual(kw.Log(10000), 2 * kw.Log(100))
        print(kw.Log(x) * kw.Log(y))
        print(kw.Log(x) * kw.Log(x))

    def test_divide(self):
        self.assertEqual(kw.Log(10000) / kw.Log(100), kw.Log(100))
        x = kw.Var('x')
        y = kw.Var('y')
        print(kw.Log(x) / kw.Log(y))
        print(kw.Log(x) / kw.Log(100))
        print(kw.Log(x) / 2)

    def test_power(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual(kw.Log(x * y) ** 2, kw.Log("log(xy)^2"))
        self.assertEqual(kw.Log(100) ** 2, 4)

    def test_assign(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual(kw.Log(x).when(x=5), kw.Log(5))
        self.assertEqual(kw.Log(x ** 2 + 2 * x * y + y ** 2).when(y=5), kw.Log(x ** 2 + 10 * x + 25))

    def test_str(self):
        x = kw.Var('x')
        print(kw.Log(x ** 2 + 6 * x + 8))
        print(-3 * kw.Log(kw.Sin(x ** 2), 5))

    def test_python_syntax(self):
        x = kw.Var('x')
        print(kw.Log(x ** 2 + 6 * x + 8).python_syntax())
        print(-3 * kw.Log(kw.Sin(x ** 2), 5).python_syntax())


class TestAbs(unittest.TestCase):
    def test_init(self):
        x = kw.Var('x')
        self.assertEqual(kw.Abs(x), kw.Abs(x))

    def test_add(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.Abs(x) + kw.Abs(x), 2 * kw.Abs(x))
        self.assertEqual(kw.Abs(y) + kw.Abs(x) == kw.Abs(x) + kw.Abs(y))

    def test_sub(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.Abs(x) - kw.Abs(x), 0)
        self.assertEqual(2 * kw.Abs(x) - kw.Abs(x), kw.Abs(x))
        self.assertEqual(2 * kw.Abs(x) ** 2 - kw.Abs(x) ** 2, kw.Abs(x) ** 2)

    def test_try_evaluate(self):
        x = kw.Var('x')
        self.assertEqual(kw.Abs(5).try_evaluate(), 5)
        self.assertEqual(kw.Abs(-5).try_evaluate(), 5)
        self.assertEqual(kw.Abs(-7).try_evaluate(), kw.Abs(7).try_evaluate())
        self.assertEqual(kw.Abs(expression=x, power=0, coefficient=5), 5)

    def test_multiply(self):
        x = kw.Var('x')
        self.assertEqual(kw.Abs(x) * 5, 5 * kw.Abs(x))
        self.assertEqual(kw.Abs(3) * kw.Abs(-3), kw.Abs(-9))
        self.assertEqual(kw.Abs(x) * kw.Abs(-3), 3 * kw.Abs(x))
        self.assertEqual(x * kw.Abs(x), x * kw.Abs(x))

    def test_divide(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.Abs(x) / 5, 0.2 * kw.Abs(x))
        self.assertEqual(x * kw.Abs(x) / x, kw.Abs(x))
        self.assertEqual(kw.Abs(x) / kw.Abs(x), 1)
        self.assertEqual(kw.Abs(x) ** 2 / kw.Abs(x), kw.Abs(x))
        self.assertEqual((3 * kw.Abs(x) ** 3) / (2 * kw.Abs(x)), 1.5 * kw.Abs(x) ** 2)

    def test_assign(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.Abs(x).when(x=5), kw.Abs(5))
        self.assertEqual(kw.Abs(x + y).when(x=5, y=5), 10)
        self.assertEqual((x * kw.Abs(x)).when(x=5), 25)
        self.assertEqual((y * kw.Abs(x)).when(y=3), 3 * kw.Abs(x))
        self.assertEqual((y * kw.Abs(x)).when(x=3), 3 * y)

    def test_to_lambda(self):
        x, y = kw.Var('x'), kw.Var('y')
        first_lambda = (kw.Abs(x).to_lambda())
        self.assertEqual(first_lambda(-5), 5)
        second_lambda = (kw.Abs(x + y)).to_lambda()
        self.assertEqual(second_lambda(-5, -4), 9)


class TestRoot(unittest.TestCase):
    def test_init(self):
        x, y = kw.Var('x'), kw.Var('y')
        print(kw.Root(x ** 2 + 2 * x + 7))
        print(kw.Root(kw.Sin(x)))

    def test_add(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.Root(x) + kw.Root(x), 2 * kw.Root(x))
        self.assertEqual(kw.Root(x) + kw.Root(y), kw.Root(y) + kw.Root(x))
        self.assertEqual(kw.Root(4) + kw.Root(4), kw.Root(16))

    def test_sub(self):
        x, y = kw.Var('x'), kw.Var('y')
        print(kw.Root(x) - kw.Root(x))
        self.assertEqual(kw.Root(x) - kw.Root(x), 0)
        self.assertEqual(3 * kw.Root(x) - 2 * kw.Root(x), kw.Root(x))

    def test_mul(self):
        pass

    def test_divide(self):
        pass

    def test_try_evaluate(self):
        self.assertEqual(kw.Root(8).try_evaluate(), math.sqrt(8))

    def test_equal(self):
        pass


class TestFraction(unittest.TestCase):

    def test_init(self):
        a = kw.Var('a')
        b = kw.Var('b')
        self.assertEqual(kw.Fraction(a, b), a / b)
        print(kw.Fraction(kw.Sin(a), kw.Sin(b)))

    def test_add(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual(x / y + x / y, 2 * x / y)
        self.assertEqual(x / y + x / (2 * y), 1.5 * x / y)

    def test_sub(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual((x / y - x / y) == 0, True)
        self.assertEqual(3 * x / y - x / y, 2 * x / y)
        self.assertEqual(x / y + x / (2 * y), 1.5 * x / y)

    def test_mul(self):
        x, y = kw.Var('x'), kw.Var('x')
        self.assertEqual(x / y * x / y, x ** 2 / y ** 2)
        self.assertEqual(x / y * x, x ** 2 / y)
        self.assertEqual(x / y * 3, 3 * x / y)

    def test_divide(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual((x / y) / (x / y), 1)
        self.assertEqual((2 * x / y) / (x / y), 2)

    def test_try_evaluate(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.Fraction(3, 4).try_evaluate(), 0.75)
        self.assertEqual(kw.Fraction(2, 3).try_evaluate(), 2 / 3)
        self.assertEqual(kw.Fraction(0, x).try_evaluate(), 0)
        self.assertEqual(kw.Fraction(x ** 2 + 2 * x * y + y ** 2, (x + y) ** 2).try_evaluate(), 1)

    def test_assign(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual((kw.Sin(x) / y).when(x=0.5 * math.pi, y=2), kw.Fraction(kw.Sin(0.5 * math.pi), 2))
        self.assertEqual((x / y).when(y=5), kw.PolyFraction(x / 5))

    def test_equal(self):
        x, y = kw.Var('x'), kw.Var('y')
        print(x / y * y / x)
        self.assertEqual(x / y * y / x == 1, True)


class TestVector(unittest.TestCase):
    def test_add(self):
        self.assertEqual(kw.Vector([1, 2, 4]) + kw.Vector([5, 2, 7]), kw.Vector([6, 4, 11]))
        self.assertEqual(kw.Vector([1, 2]) + kw.Vector([5, 2]), kw.Vector([6, 4]))

    def test_sub(self):
        self.assertEqual(kw.Vector([1, 2, 4]) - kw.Vector([5, 2, 7]), kw.Vector([-4, 0, -3]))
        self.assertEqual(kw.Vector([1, 2]) - kw.Vector([5, 2]), kw.Vector([-4, 0]))

    def test_eq(self):
        self.assertEqual(kw.Vector([4, 3, 1]) == kw.Vector([8, 6, 1]), False)
        self.assertEqual(kw.Vector([4, 3, 1]) == kw.Vector((4, 3, 1)), True)
        self.assertEqual(kw.Vector([4, 3, 1]) == kw.Vector([4, 3, 1], start_coordinate=[7, 4, 2]), True)

    def test_multiply(self):
        self.assertEqual(kw.Vector([1, 2]) * kw.Vector([5, 3]), 11)
        self.assertEqual(kw.Vector([5, 2, 6]) * kw.Vector([3, 4, 1]), 29)

    def test_plot(self):
        my_2d_vector = kw.Vector2D(5, 4)
        my_2d_vector.plot()
        my_3d_vector = kw.Vector3D(4, 6, 8)
        my_3d_vector.plot()


class TestFunction(unittest.TestCase):
    def test_init(self):
        x = kw.Var('x')
        self.assertEqual(kw.Function("f(x) = 2x"), kw.Function("x => 2x"))
        self.assertEqual(kw.Function(x ** 2 + 3 * x + 7), kw.Function("f(x) = x^2 + 3x + 7"))
        self.assertEqual(kw.Function("x,y => x+y"), kw.Function("f(m,n) = m+n"))

    def test_compute(self):
        self.assertEqual(kw.Function("f(x) = 2x")(4), 8)
        self.assertEqual(kw.Function("f(x) = x^2")(4), 16)
        self.assertEqual(kw.round_decimal(kw.Function("f(x) = sin(x)")(math.pi)), 0)
        self.assertEqual(kw.Function("f(x) = ln(x)")(math.e), 1)
        self.assertEqual(kw.Function("f(x,y) = x+y")(1, 4), 5)

    def test_toIExpression(self):
        x = kw.Var('x')
        y = kw.Var('y')
        self.assertEqual(kw.Function("f(x) = x^2 + 6x + 8").toIExpression(), x ** 2 + 6 * x + 8)
        self.assertEqual(kw.Function("f(x) = sin(x) + cos(x)").toIExpression(), kw.Sin(x) + kw.Cos(x))
        self.assertEqual(kw.Function("f(x,y) = x+y").toIExpression(), x + y)
        self.assertEqual(kw.Function("f(x) = sin(x)*cos(y)").toIExpression(), kw.Sin(x) * kw.Cos(y))
        # self.assertEqual(kw.Function("f(x) = sin(ln(x))").toIExpression(), kw.Sin(kw.Ln(x))) ** NOT SUPPORTED YET **

    def test_y_intersection(self):
        self.assertEqual(kw.Function("f(x) = 2x + 5").y_intersection(), 5)
        self.assertEqual(kw.Function("f(x) = x^2 + 6x +7").y_intersection(), 7)
        self.assertEqual(kw.Function("f(x) = sin(x)").y_intersection(), 0)

    def test_chain(self):
        my_function = kw.Function("f(x) = 2x").chain("f(x) = x+5")
        other_function = kw.Function("f(x) = 2x").chain("f(x) = x^2")

    def test_plot(self):
        kw.Function("f(x) = 2x*sin(x)").plot()
        kw.Function("f(x,y) = sin(x)*cos(y)").plot()

    def test_scatter(self):
        kw.Function("f(x) = x**2").scatter2d()
        kw.Function("f(x,y) = sin(x) * cos(y) + ln(x)").scatter3d()
        kw.Function("f(x) = x^2").scatter()
        kw.Function("f(x,y) = x+y").scatter()


class TestExpressionMul(unittest.TestCase):
    def test_init(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.ExpressionMul([3 * x, 2 * y]), kw.ExpressionMul([2 * y, 3 * x]))  # implement here


class TestExpressionSum(unittest.TestCase):
    def test_init(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.Abs(x) + kw.Root(y), kw.Root(y) + kw.Abs(x))
        self.assertEqual(kw.Sin(2*x) + kw.Abs(-4*y), kw.Abs(-4*y) + kw.Sin(2*x))
        self.assertEqual(kw.ExpressionSum([kw.Sin(x), kw.Log(x)]), kw.Log(x) + kw.Sin(x))

    def test_add(self):
        x, y = kw.Var('x'), kw.Var('y')
        my_expressions = kw.ExpressionSum([kw.Abs(x), kw.Sin(x)])
        # self.assertEqual(my_expressions + kw.Abs(x), 2*kw.Abs(x) + kw.Sin(x)) # IMPROVE LATER
        print(my_expressions + kw.Log(x))
        print(kw.Log(x) + my_expressions)
        self.assertEqual(my_expressions + kw.Log(x), kw.Log(x) + my_expressions)
        self.assertEqual(my_expressions + kw.Log(x), kw.Log(x) + kw.Sin(x) + kw.Abs(x))

    def test_sub(self):
        x, y = kw.Var('x'), kw.Var('y')
        my_expressions = kw.ExpressionSum([kw.Abs(x), kw.Sin(x)])
        self.assertEqual(my_expressions - kw.Log(x), - kw.Log(x) + kw.Sin(x) + kw.Abs(x))

    def test_mul(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.ExpressionSum([kw.Sin(x), kw.Abs(x)]) * 5, 5*kw.Sin(x) + 5*kw.Abs(x))
        self.assertEqual(kw.ExpressionSum([kw.Sin(x), kw.Abs(x)]) * kw.Sin(x), kw.Sin(x)**2 + kw.Abs(x)*kw.Sin(x))

    def test_div(self):
        x, y = kw.Var('x'), kw.Var('y')
        self.assertEqual(kw.ExpressionSum([kw.Sin(x), kw.Abs(x)]) / 5, 0.2 * kw.Sin(x) + 0.2 * kw.Abs(x))
        print(kw.ExpressionSum([kw.Sin(x), kw.Abs(x)]) / kw.Sin(x))
        print(1 + kw.Abs(x)/kw.Sin(x))
        self.assertEqual(kw.ExpressionSum([kw.Sin(x), kw.Abs(x)]) / kw.Sin(x), 1 + kw.Abs(x)/kw.Sin(x))

    def test_pow(self):
        pass

    def test_tryEvaluate(self):
        self.assertEqual(kw.ExpressionSum([kw.Abs(-5), kw.Ln(kw.e)]).try_evaluate(), 6)

    def test_simplify(self):
        x = kw.Var('x')
        my_expressions = kw.ExpressionSum([kw.Sin(x), kw.Ln(x), 0])
        my_expressions.simplify()
        self.assertEqual(my_expressions, kw.Sin(x) + kw.Ln(x))

        my_expressions = kw.ExpressionSum([kw.Sin(x), kw.Ln(x), kw.Ln(math.e)])
        my_expressions.simplify()
        self.assertEqual(my_expressions, kw.Sin(x) + kw.Ln(x) + 1)

class EquationSolving(unittest.TestCase):
    def test_solveLinear(self):
        self.assertEqual(kw.solve_linear("3x+5=8"), 1)
        self.assertEqual(kw.solve_linear("2x=2x"), np.inf)
        self.assertEqual(kw.solve_linear("0=5"), None)

    def test_solve_quadratic(self):
        self.assertEqual(kw.solve_quadratic(1, 6, 8), (-2, -4))
