# built in imports
from typing import List, Iterable, Union, Optional 
import warnings
import json
from math import comb
import random


# kiwicalc imports
from plot.models import IExpression, IPlottable
import mono
import var
from ....kiwicalc.plot.plot import plot_function, plot_functions_3d 
from ....kiwicalc.equations.numerical import * 
from ....kiwicalc.string_analysis import to_lambda 

# TODO: try not to use the "from" syntax for circular imports

from string_analysis import poly_from_str


class Poly(IExpression, IPlottable):
    __slots__ = ['_expressions', '__loop_index']

    def __init__(self, expressions):
        self.__loop_index = 0
        if isinstance(expressions, str):
            self._expressions: List[mono.Mono] = poly_from_str(
                expressions, get_list=True)
            self.simplify()
        elif isinstance(expressions, (int, float)):
            self._expressions = [mono(expressions)]
        elif isinstance(expressions, mono.Mono):
            self._expressions = [expressions.__copy__()]
        elif isinstance(expressions, Iterable):
            self._expressions = []
            for expression in expressions:
                if isinstance(expression, mono.Mono):
                    # avoiding memory sharing by passing by value
                    self._expressions.append(expression.__copy__())
                elif isinstance(expression, str):
                    self._expressions += (
                        poly_from_str(expression, get_list=True))
                elif isinstance(expression, Poly):
                    self._expressions.extend(expression.expressions.copy())
                elif isinstance(expression, (int, float)):
                    self._expressions.append(mono.Mono(expression))
                else:
                    warnings.warn(
                        f"Couldn't process expression '{expression} with invalid type {type(expression)}'")
            self.simplify()
        elif isinstance(expressions, Poly):
            self._expressions: List[mono.Mono] = [
                mono_expression.__copy__() for mono_expression in expressions._expressions]
            self.simplify()

        elif isinstance(expressions, mono.Mono):
            self._expressions: List[mono.Mono] = [expressions.__copy__()]
        else:
            raise TypeError(
                f"Invalid type {type(expressions)} in Poly.__init__(). Allowed types: list,tuple,Mono,"
                f"Poly,str,int,float "
            )

    @property
    def expressions(self):
        return self._expressions

    @expressions.setter
    def expressions(self, expressions):
        self._expressions = expressions

    def __iadd__(self, other: Union[IExpression, int, float, str]):
        if other == 0:
            return self
        if isinstance(other, (int, float)):
            self.__add_mono.Monomial(mono.Mono(other))
            if all(expression.coefficient == 0 for expression in self.expressions):
                self.expressions = [mono.Mono(0)]
            self.simplify()
            return self

        elif isinstance(other, str):
            expressions = poly_from_str(other, get_list=True)
            for mono.Mono_expression in expressions:
                self.__add_monomial(mono.Mono_expression)
            if all(expression.coefficient == 0 for expression in self.expressions):
                self.expressions = [mono.Mono(0)]
            self.simplify()
            return self

        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                self.__add_monomial(mono.Mono(other_evaluation))
                self.simplify()
                return self

            if isinstance(other, mono.Mono):
                self.__add_monomial(other)
                if all(expression.coefficient == 0 for expression in self.expressions):
                    self.expressions = [Mono(0)]
                self.simplify()
                return self

            elif isinstance(other, Poly):
                for mono.Mono_expression in other.expressions:
                    self.__add_monomial(mono_expression)
                if all(expression.coefficient == 0 for expression in self.expressions):
                    self.expressions = [mono.Mono(0)]
                self.simplify()
                return self
            else:  # If it's just a random ExpressionSum expression
                return ExpressionSum((self, other))
        else:
            raise TypeError(
                f"__add__ : invalid type '{type(other)}'. Allowed types: str, Mono, Poly, int, or float"
            )

    def __add_monomial(self, other: mono.Mono) -> None:
        self.__filter_zeroes()
        for index, expression in enumerate(self.expressions):
            if expression.variables_dict == other.variables_dict or (not expression.variables and not other.variables):
                # if they can be added
                self._expressions[index] += other
                return
        self._expressions.append(other)

    def __sub_monomial(self, other: mono.Mono) -> None:
        self.__filter_zeroes()
        for index, expression in enumerate(self._expressions):
            if expression.variables_dict == other.variables_dict or (not expression.variables and not other.variables):
                # if they can be subtracted
                self._expressions[index] -= other
                return  # Break out of the function.

        self.expressions.append(-other)

    def __rsub__(self, other: Union[int, float, str, IExpression]):
        if isinstance(other, (int, float, str)):
            other = Poly(other)

        if isinstance(other, mono.Mono):
            return Poly([expression / other for expression in self.expressions])
        elif isinstance(other, Poly):
            other.__isub__(self)
            return other
        elif isinstance(other, IExpression):
            return ExpressionMul((other, -self))
        else:
            raise TypeError(
                f"Poly.__rsub__: Expected types int,float,str,Mono,Poly, but got {type(other)}")

    def __isub__(self, other: Union[int, float, IExpression, str]):
        if isinstance(other, (int, float)):
            self.__sub_monomial(mono.Mono(other))
            if all(expression.coefficient == 0 for expression in self.expressions):  # TODO: check if needed
                self.expressions = [mono.Mono(0)]
            self.simplify()
            return self

        elif isinstance(other, str):
            expressions = poly_from_str(other, get_list=True)
            for mono_expression in expressions:
                self.__sub_monomial(mono_expression)
            if all(expression.coefficient == 0 for expression in self.expressions):
                self.expressions = [mono.Mono(0)]
            self.simplify()
            return self

        elif isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                self.__sub_monomial(mono.Mono(other_evaluation))
                self.simplify()
                return self

            if isinstance(other, mono.Mono):
                self.__sub_monomial(other)
                if all(expression.coefficient == 0 for expression in self._expressions):
                    self.expressions = [mono.Mono(0)]
                self.simplify()
                return self

            elif isinstance(other, Poly):
                for mono_expression in other._expressions:
                    self.__sub_monomial(mono_expression)
                if all(expression.coefficient == 0 for expression in self._expressions):
                    self.expressions = [Mono(0)]
                self.simplify()
                return self
            else:  # If it's just a random IExpression expression
                return ExpressionSum((self, -other))
        else:
            raise TypeError(
                f"Invalid type '{type(other)} while subtracting polynomials.")

    def __neg__(self):
        return Poly([-expression for expression in self.expressions])

    # TODO: try to make it more efficient ..
    def __imul__(self, other: Union[int, float, IExpression]):
        if other == 0:
            return mono.Mono(coefficient=0)
        if isinstance(other, (int, float)):
            for index, expression in enumerate(self._expressions):
                self._expressions[index].coefficient *= other
            if all(expression.coefficient == 0 for expression in self._expressions):
                self._expressions = [mono.Mono(0)]
            self.simplify()
            return self
        if isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation is not None:
                for index in range(len(self._expressions)):
                    self._expressions[index] *= other_evaluation
                self.simplify()
                return self
            if isinstance(other, mono.Mono):
                for index, expression in enumerate(self._expressions):
                    self._expressions[index] *= other
                self.simplify()
                return self
            elif isinstance(other, Poly):
                new_expressions = []
                for expression1 in self.expressions:
                    for expression2 in other.expressions:

                        result = expression1 * expression2
                        found = False
                        for index, new_expression in enumerate(
                                new_expressions):  # Checking whether to append or simplify
                            if new_expression.variables_dict == result.variables_dict:
                                addition_result = new_expression + result
                                if addition_result.coefficient == 0:
                                    del new_expressions[index]
                                else:
                                    new_expressions[index] = addition_result
                                found = True
                                break
                        if not found:
                            new_expressions.append(
                                result.__copy__())  # is it necessary ?
                self._expressions = new_expressions
                self.simplify()
                return self
            else:  # Multiply by an unknown IExpression. Could be Root, Fraction, etc.
                return other * self

        elif isinstance(other, Matrix):  # Check if this works
            other.multiply_all(self)
        elif isinstance(other, Vector):
            raise NotImplementedError
        elif isinstance(other, Iterable):
            return [item * self for item in other]

    def __filter_zeroes(self):
        if len(self._expressions) > 1:
            for index, expression in enumerate(self._expressions):
                if expression.coefficient == 0:
                    del self.expressions[index]

    def divide_by_number(self, number: int):
        for mono_expression in self._expressions:
            mono_expression.divide_by_number(number)
        return self

    def divide_by_poly(self, other: "Union[Mono, Poly]", get_remainder=False, nmax=1000):
        # TODO: fix this method ....
        if isinstance(other, Poly) and len(other.expressions) == 1:
            # If the polynomial contains only one monomial, turn it to Mono
            other = other.expressions[0]
        if isinstance(other, mono.Mono):
            if other.coefficient == 0:
                raise ZeroDivisionError(
                    "cannot divide by an expression whose coefficient is zero")
            other_copy = other.__copy__()
            other_copy.coefficient = 1 / other_copy.coefficient
            if other_copy.variables_dict is not None:
                other_copy.variables_dict = {variable: -value for (variable, value) in
                                             other_copy.variables_dict.items()}
                # dividing by x^3 is equivalent to multiplying by x^-3
            if get_remainder:
                return self.__imul__(other_copy), 0
            return self.__imul__(other_copy)
        elif isinstance(other, Poly):
            new_expression, remainder = mono.Mono(0), 0
            temp_expressions = Poly(self._expressions.copy())
            for i in range(nmax):
                if len(temp_expressions._expressions) == 0:
                    new_expression.simplify()
                    if get_remainder:
                        return new_expression, 0
                    return new_expression
                if len(temp_expressions._expressions) == 1 and temp_expressions.expressions[0].variables_dict is None:
                    if get_remainder:
                        return new_expression, other._expressions[0]
                    return new_expression + other._expressions[0] / other

                first_item = temp_expressions._expressions[0] / \
                    other._expressions[0]
                new_expression += first_item.__copy__()
                subtraction_expressions = first_item * other
                temp_expressions -= subtraction_expressions
                if len(temp_expressions.expressions) == 1:
                    if temp_expressions.expressions[0].coefficient == 0:
                        if isinstance(new_expression, Poly):
                            new_expression.simplify()
                        if get_remainder:
                            return new_expression, remainder
                        if remainder == 0:
                            return new_expression
                        new_expression += Fraction(remainder, other)
                        return new_expression
                    # Reached a result with a remainder
                    elif temp_expressions.expressions[0].variables_dict is None:
                        if isinstance(new_expression, Poly):
                            new_expression.sort()
                        remainder = temp_expressions.expressions[0].coefficient
                        if get_remainder:
                            return new_expression, remainder
                        if remainder == 0:
                            return new_expression
                        new_expression += Fraction(remainder, other)
                        return new_expression
                    else:  # The remainder is algebraic
                        warnings.warn(
                            "Got an algebraic remainder when dividing Poly objects")
                        if isinstance(new_expression, Poly):
                            new_expression.sort()
                        if get_remainder:
                            return new_expression, remainder
                        if remainder == 0:
                            return new_expression
                        new_expression += Fraction(remainder, other)
                        return new_expression
            warnings.warn("Division timed out ...")
            return PolyFraction(self, other)

    def __itruediv__(self, other: Union[int, float, IExpression], get_remainder=False):
        my_evaluation = self.try_evaluate()
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("cannot divide by 0")
            if my_evaluation is None:  # if the Poly object can't be evaluated into a free number
                if get_remainder:
                    return self.divide_by_number(other), 0
                return self.divide_by_number(other)
            else:
                if get_remainder:
                    return mono.Mono(coefficient=my_evaluation / other), 0
                return mono.Mono(coefficient=my_evaluation / other)
        if isinstance(other, IExpression):
            other_evaluation = other.try_evaluate()
            if other_evaluation == 0:
                raise ZeroDivisionError(
                    f"Cannot divide a polynomial by the expression {other} which evaluates to 0")
            if None not in (my_evaluation, other_evaluation):
                # Both expressions can be evaluated
                if get_remainder:
                    return mono.Mono(coefficient=my_evaluation / other_evaluation), 0
                return mono.Mono(coefficient=my_evaluation / other_evaluation)
            elif my_evaluation is None and other_evaluation is not None:
                # only the other object can be evaluated into a number
                if get_remainder:
                    return self.divide_by_number(other_evaluation), 0
                return self.divide_by_number(other_evaluation)
            else:  # Both expressions can't be evaluated into numbers apparently
                if isinstance(other, (Poly, mono.Mono)):
                    return self.divide_by_poly(other, get_remainder=get_remainder)
                else:
                    return Fraction(self, other)
        else:
            raise TypeError(
                f"Invalid type '{type(other)} when dividing Poly objects' ")

    def __truediv__(self, other: Union[int, float, IExpression], get_remainder=False):
        return self.__copy__().__itruediv__(other, get_remainder=get_remainder)

    def __calc_binomial(self, power: int):
        """Internal method for using the newton's binomial in order to speed up calculations in the form (a+b)^2"""
        expressions = []
        first, second = self._expressions[0], self._expressions[1]
        if_number1, if_number2 = first.variables_dict is None, second.variables_dict is None
        for k in range(power + 1):
            comb_result = comb(power, k)
            first_power, second_power = power - k, k
            if if_number1:
                first_expression = mono.Mono(
                    first.coefficient ** first_power * comb_result)
            else:
                first_expression = mono.Mono(first.coefficient ** first_power * comb_result,
                                             {key: value * first_power for (key, value) in
                                              first.variables_dict.items()})
            if if_number2:
                second_expression = mono.Mono(
                    second.coefficient ** second_power)
            else:
                second_expression = mono.Mono(second.coefficient ** second_power,
                                              {key: value * second_power for (key, value) in
                                               second.variables_dict.items()})
            expressions.append(first_expression * second_expression)
        return Poly(expressions)

    def __pow__(self, power: Union[int, float, IExpression, str], modulo=None):
        if isinstance(power, float):  # Power by float is not supported yet ...
            power = int(power)
        if not isinstance(power, int):
            if isinstance(power, str):
                power = Poly(power)
            if isinstance(power, mono.Mono):
                if power.variables_dict is not None:
                    raise ValueError(
                        "Cannot perform power with an algebraic exponent on polynomials")
                    # TODO: implement exponents for that
                else:
                    power = power.coefficient
            elif isinstance(power, Poly):
                if len(power._expressions) == 1 and power._expressions[0].variables_dict is None:
                    power = power._expressions[0].coefficient
                else:
                    raise ValueError(
                        "Cannot perform power with an algebraic exponent")
        if power == 0:
            return Poly(1)
        elif power == 1:
            return Poly(self._expressions)

        my_evaluation = self.try_evaluate()
        if my_evaluation is not None:
            return mono.Mono(coefficient=my_evaluation ** power)

        if len(self.expressions) == 2:  # FOR TWO ITEMS, COMPUTE THE RESULT WITH THE BINOMIAL THEOREM
            return self.__calc_binomial(power)

        # FOR MORE THAN TWO ITEMS, OR JUST 1, CALCULATE IT AS MULTIPLICATION ( LESS EFFICIENT )
        else:
            new_expression = self
            for i in range(power - 1):
                new_expression *= self
            return new_expression

    # TODO: for that, basic exponents need to be implemented.
    def __rpow__(self, other, power, modulo=None):
        if len(self._expressions) == 1 and self._expressions[0].variables_dict is None:
            if not isinstance(other, (mono.Mono, Poly)):
                other = Poly(other)
            return other.__pow__(self)

        else:
            return Exponent(self, other)

    def __ipow__(self, other):  # TODO: re-implement it later
        self._expressions = self.__pow__(other)._expressions
        return self

    def is_number(self):
        return all(expression.is_number() for expression in self._expressions)

    def try_evaluate(self) -> Optional[Union[int, float, complex]]:
        if not self._expressions:
            return 0
        if self.is_number() and (length := len(self._expressions)) > 0:
            if length > 1:
                self.simplify()
            return self._expressions[0].coefficient
        return None

    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(other, str):
            other = Poly(other)
        if isinstance(other, (int, float, mono.Mono)):
            if len(self._expressions) != 1:
                return False
            return self._expressions[0].__eq__(other)
        elif isinstance(other, Poly):
            self.simplify()
            other.simplify()
            my_num_of_variables = self.num_of_variables
            other_num_of_variables = other.num_of_variables
            if my_num_of_variables != other_num_of_variables:
                return False
            if my_num_of_variables == 0:  # both expressions don't contain any variable, meaning only free numbers
                if len(self._expressions) != len(other._expressions):
                    return False
                return self._expressions[0] == other._expressions[
                    1]  # After simplification,only one free number is left
            elif my_num_of_variables == 1:  # both expressions have one variable
                # all items should be in the same place when sorted!
                return self._expressions == other._expressions
            else:  # more than one variable
                expressions_checked = []
                for expression in self._expressions:
                    if expression not in expressions_checked:
                        instances_in_other = other._expressions.count(
                            expression)
                        instances_in_me = self._expressions.count(expression)
                        if instances_in_other != instances_in_me:
                            return False
                        expressions_checked.append(expression)

                for other_expression in other._expressions:
                    if other_expression not in expressions_checked:
                        instances_in_me = self._expressions.count(
                            other_expression)
                        instances_in_other = other._expressions.count(
                            other_expression)
                        if instances_in_me != instances_in_other:
                            return False
                        expressions_checked.append(expression)
            return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        self.__loop_index = 0
        return self

    def __next__(self):
        if self.__loop_index < len(self.expressions):
            result = self._expressions[self.__loop_index]
            self.__loop_index += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self._expressions.__getitem__(item)

    def derivative(self):
        for expression in self._expressions:
            if expression.variables_dict is not None and len(expression.variables_dict) > 1:
                raise ValueError(
                    "Try using partial_derivative() for expression with more than variable")
        derived_expression = Poly([expression.derivative()
                                  for expression in self._expressions])
        derived_expression.simplify()
        derived_expression.sort()
        return derived_expression

    def is_empty(self) -> bool:
        return not self._expressions

    # TODO: implement it more efficiently
    def partial_derivative(self, variables: Iterable):
        derived_expression = Poly((monomial.partial_derivative(
            variables) for monomial in self._expressions))
        derived_expression.simplify()
        derived_expression.sort()
        if derived_expression.is_empty():
            return mono.Mono(0)
        return derived_expression

    def integral(self, add_c=False):
        for expression in self.expressions:
            if expression.variables_dict is not None and len(expression.variables_dict) > 1:
                raise ValueError(
                    f"IExpression {expression.__str__()}: Can only compute the integral with one variable or "
                    f"less ( got {len(expression.variables_dict)}")
        # TODO: fix double values problem.
        result = Poly([expression.integral()
                      for expression in self.expressions])
        if add_c:
            c = var.Var('c')
            result += c
        return result

    @property
    def variables(self):
        variables = set()
        for expression in self._expressions:
            variables.update(expression.variables)
        return variables

    @property
    def num_of_variables(self):
        return len(self.variables)

    def coefficients(self):
        """
        convert the polynomial expression to a list of coefficients. Currently works only with one variable.
        :return:
        """
        number_of_variables = self.num_of_variables
        if number_of_variables == 0:  # No variables_dict found - free number or empty expression
            num_of_expressions = len(self._expressions)
            if num_of_expressions == 0:
                return None
            elif num_of_expressions == 1:
                return [self._expressions[0].coefficient]
            elif num_of_expressions > 1:
                self.simplify()
                return [self._expressions[0].coefficient]
        elif number_of_variables > 1:  # Expressions with more than one variable aren't valid!
            raise ValueError(
                f"Can only fetch the coefficients of a polynomial with 1 variable, found {number_of_variables}")
        # One variable - expressions such as x^2 + 6
        sorted_exprs = sorted_expressions(
            [expression for expression in self._expressions if expression.variables_dict is not None])
        biggest_power = max_power(sorted_exprs)
        coefficients = [0] * \
            (int(fetch_power(biggest_power.variables_dict)) + 1)
        for index, sorted_expression in enumerate(sorted_exprs):
            coefficients[
                len(coefficients) - int(
                    fetch_power(sorted_expression.variables_dict)) - 1] = sorted_expression.coefficient
        free_numbers = [
            expression for expression in self._expressions if expression.variables_dict is None]
        free_number = sum(
            (expression.coefficient for expression in free_numbers))
        coefficients[-1] = free_number
        return coefficients

    def assign(self, **kwargs):  # TODO: check if it's even working !
        for expression in self._expressions:
            expression.assign(**kwargs)
        self.simplify()

    def discriminant(self):
        my_coefficients = self.coefficients()
        length = len(my_coefficients)
        if length == 1:  # CONSTANT
            # There is no common convention for a discriminant of a constant polynomial.
            return 0
        elif length == 2:  # LINEAR - convention is 1
            return 1
        elif length == 3:  # QUADRATIC
            return my_coefficients[1] ** 2 - 4 * my_coefficients[0] * my_coefficients[2]
        elif length == 4:  # CUBIC
            # depressed cubic : x^3 + px + q
            if my_coefficients[0] == 1 and my_coefficients[1] == 0:
                return -4 * my_coefficients[2] ** 3 - 27 * my_coefficients[3] ** 2
        elif length == 5:  # QUARTIC
            a, b, c, d, e = my_coefficients[0], my_coefficients[1], my_coefficients[2], my_coefficients[3], \
                my_coefficients[4]
            result = 256 * a ** 3 * e ** 3 - 192 * a ** 2 * b * d * e ** 2 - \
                128 * a ** 2 * c ** 2 * e ** 2 + 144 * a ** 2 * c * d ** 2 * e
            result += -27 * a ** 2 * d ** 4 + 144 * a * b ** 2 * c * e ** 2 - \
                6 * a * b ** 2 * d ** 2 * e - 80 * a * b * c ** 2 * d * e
            result += 18 * a * b * c * d ** 3 + 16 * a * c ** 4 * e - 4 * a * \
                c ** 3 * d ** 2 - 27 * b ** 4 * e ** 2 + 18 * b ** 3 * c * d * e
            result += -4 * b ** 3 * d ** 3 - 4 * b ** 2 * \
                c ** 3 * e + b ** 2 * c ** 2 * d ** 2
            return result
        else:
            raise ValueError(
                "Discriminants are not supported yet for polynomials with degree 5 or more")

    def roots(self, epsilon=0.000001, nmax=10_000):
        my_coefficients = self.coefficients()
        return solve_polynomial(my_coefficients, epsilon, nmax)

    def real_roots(self):
        pass

    def extremums(self):
        num_of_variables = len(self.variables)
        if num_of_variables == 0:
            return None
        elif num_of_variables == 1:
            my_lambda = self.to_lambda()
            my_derivative = self.derivative()
            if my_derivative.is_number():
                return None
            derivative_roots = my_derivative.roots(nmax=1000)
            myRoots = [Point2D(root.real, my_lambda(root.real))
                       for root in derivative_roots if root.imag <= 0.00001]
            return PointCollection(myRoots)

    def extremums_axes(self, get_derivative=False):
        num_of_variables = len(self.variables)
        if num_of_variables == 0:
            return None
        elif num_of_variables == 1:
            my_derivative = self.derivative()
            if my_derivative.is_number():
                return None
            my_roots = [root.real for root in my_derivative.roots(
                nmax=1000) if root.imag <= 0.00001]
            my_roots.sort()
            if get_derivative:
                return my_roots, my_derivative
            return my_roots

    def up_and_down(self):
        extremums_axes, my_derivative = self.extremums_axes(
            get_derivative=True)
        return self.__up_and_down(extremums_axes, my_derivative)

    def __up_and_down(self, extremums_axes, my_derivative=None):
        x = var.Var('x')
        coefficients = self.coefficients()
        num_of_coefficients: int = len(coefficients)
        if num_of_coefficients == 1:  # free number
            return None, None  # the function just stays constant
        elif num_of_coefficients == 2:  # linear function
            if coefficients[0] > 0:
                return Range(expression=x, limits=(-np.inf, np.inf), operators=(LESS_THAN, LESS_THAN)), None
            elif coefficients[0] < 0:
                return None, Range(expression=x, limits=(-np.inf, np.inf), operators=(LESS_THAN, LESS_THAN))
        elif num_of_coefficients == 2:  # Quadratic function
            first = Range(expression=x, limits=(-np.inf,
                          extremums_axes[0]), operators=(LESS_THAN, LESS_THAN))
            second = Range(expression=x, limits=(
                extremums_axes[0], np.inf), operators=(LESS_THAN, LESS_THAN))
            if coefficients[0] > 0:  # Happy parabola
                return second, first
            return first, second  # Sad parabola:

        else:
            num_of_extremums = len(extremums_axes)
            if num_of_extremums == 0:
                print("didn't find any extremums...")

            if my_derivative is None:
                my_derivative = self.derivative()
            derivative_lambda = my_derivative.to_lambda()
            up_ranges, down_ranges = [], []
            derivatives_values = [
                derivative_lambda(random.uniform(extremums_axes[i], extremums_axes[i + 1])) for i in
                range(num_of_extremums - 1)]
            before_value = derivative_lambda(extremums_axes[0] - 1)
            after_value = derivative_lambda(extremums_axes[-1] + 1)
            derivatives_values.append(after_value)
            if before_value > 0:
                up_ranges.append(
                    Range(expression=x, limits=(-np.inf, extremums_axes[0]), operators=(LESS_THAN, LESS_THAN)))
            elif before_value < 0:
                down_ranges.append(
                    Range(expression=x, limits=(-np.inf, extremums_axes[0]), operators=(LESS_THAN, LESS_THAN)))
            else:
                pass

            for i in range(num_of_extremums - 1):
                random_value = derivative_lambda(random.uniform(
                    extremums_axes[i], extremums_axes[i + 1]))
                if random_value > 0:
                    up_ranges.append(
                        Range(expression=x, limits=(extremums_axes[i], extremums_axes[i + 1]),
                              operators=(LESS_THAN, LESS_THAN)))
                elif random_value < 0:
                    down_ranges.append(
                        Range(expression=x, limits=(extremums_axes[i], extremums_axes[i + 1]),
                              operators=(LESS_THAN, LESS_THAN)))
                else:
                    pass

            if after_value > 0:
                up_ranges.append(
                    Range(expression=x, limits=(extremums_axes[-1], np.inf), operators=(LESS_THAN, LESS_THAN)))
            elif after_value < 0:
                down_ranges.append(
                    Range(expression=x, limits=(extremums_axes[-1], np.inf), operators=(LESS_THAN, LESS_THAN)))
            else:
                pass

            return RangeOR(up_ranges), RangeOR(down_ranges)

    def data(self, no_roots=False):
        """
        Get a dictionary that provides information about the polynomial: string, degree, coefficients, roots, extremums, up and down.
        """
        variables = self.variables
        num_of_variables = len(variables)
        my_eval = self.try_evaluate()
        if num_of_variables == 0:
            return {
                "string": self.__str__(),
                "variables": variables,
                "plotDimensions": num_of_variables + 1,
                "coefficients": [my_eval],
                "roots": np.inf if my_eval == 0 else [],
                "y_intersection": my_eval,
                "extremums": [],
                "up": None,
                "down": None,
            }
        elif num_of_variables == 1:
            extremums_axes = self.extremums_axes()
            my_lambda = self.to_lambda()
            my_extremums = [Point2D(x, my_lambda(x)) for x in extremums_axes]
            my_derivative = self.derivative()
            up, down = self.__up_and_down(
                extremums_axes, my_derivative=my_derivative)
            return {
                "string": self.__str__(),
                "variables": variables,
                "plotDimensions": num_of_variables + 1,
                "coefficients": self.coefficients(),
                "roots": [] if no_roots else self.roots(),
                "y_intersection": my_lambda(0),
                "derivative": my_derivative,
                "extremums": my_extremums,
                "up": up.__str__(),
                "down": down.__str__()
            }
        else:
            return {
                "string": self.__str__(),
                "variables": variables,
                "plotDimensions": num_of_variables + 1,
            }

    def get_report(self, colored=True) -> str:
        if colored:
            accumulator = ""
            for key, value in self.data().items():
                accumulator += f"\033[93m{key}\33[0m: {value.__str__()}\n"
            return accumulator
        return "\n".join(value.__str__() for key, value in self.data().items())

    def _format_report(self, data):
        accumulator = [f"Function: {data['string']}"]
        variables = ", ".join(variable for variable in data["variables"])
        accumulator.append(f"variables: {variables}")
        if len(data['variables']) == 1:
            accumulator.append(f"coefficients: {data['coefficients']}")
            roots = list(data['roots'])
            for index, root in enumerate(roots):
                if isinstance(root, complex):
                    if root.imag < 0.0001:
                        roots[index] = round(root.real, 3)
            roots_string = ", ".join(str(root) for root in roots)
            accumulator.append(f"roots: {roots_string}")
            accumulator.append(
                f"Intersection with the y axis: {round(data['y_intersection'], 3)}")
            accumulator.append(f"Derivative: {data['derivative']}")
            accumulator.append("Extremums Points:" + ",".join(extremum.__str__()
                               for extremum in data['extremums']))
            accumulator.append(f"Up: {data['up']}")
            accumulator.append(f"Down: {data['down']}")

        return accumulator

    def print_report(self):
        print(self.get_report())

    def export_report(self, path: str, delete_image=True):
        c = Canvas(path)
        c.setFont('Helvetica-Bold', 22)

        c.drawString(50, 800, "Function Report")
        textobject = c.beginText(2 * cm, 26 * cm)
        c.setFont('Helvetica', 16)
        data = self.data()
        variables = ",".join(data['variables'])
        for line in self._format_report(data):
            textobject.textLine(line)
            textobject.textLine("")
        c.drawText(textobject)
        if len(variables) == 1:
            plot_function(f"f({variables}) = {data['string']}", show=False)
        else:
            plot_function_3d(f"f({variables}) = {data['string']}", show=False)
        # Long path so it won't collide with the user's images accidentally.
        plt.savefig("tempPlot1146151.png")
        if len(data['variables']) == 1 or len(data['variables']) == 2:
            if len(data['variables']) == 1:
                c.drawInlineImage("tempPlot1146151.png", 50, -
                                  215, width=500, preserveAspectRatio=True)
            elif len(data['variables']) == 2:
                c.drawInlineImage("tempPlot1146151.png", 50,
                                  200, width=500, preserveAspectRatio=True)
            if delete_image:
                os.remove("tempPlot1146151.png")
        c.showPage()
        c.save()

    def durand_kerner(self):
        return durand_kerner(self.to_lambda(), self.coefficients())

    def ostrowski(self, initial_value: float, epsilon=0.00001, nmax=10_000):
        return ostrowski_method(self.to_lambda(), self.derivative().to_lambda(), initial_value, epsilon, nmax)

    def laguerres(self, x0: float, epsilon=0.00001, nmax=100000):
        my_derivative = self.derivative()
        second_derivative = self.derivative().to_lambda()
        return laguerre_method(self.to_lambda(), my_derivative.to_lambda(), second_derivative, x0, epsilon, nmax)

    def halleys(self, initial_value=0, epsilon=0.00001, nmax=10_000):  # TODO: check if works
        """
        Halley's method is a root finding method developed by Edmond Halley for functions with continuous second
        derivatives and a single variable.
        :param initial_value:
        :param epsilon:
        :return:
        """
        f_0 = self
        f_1 = f_0.derivative()
        f_2 = f_1.derivative()

        f_0 = self.to_lambda()
        f_1 = f_1.to_lambda()
        f_2 = f_2.to_lambda()
        return halleys_method(f_0, f_1, f_2, initial_value, epsilon, nmax)

    def newton(self, initial_value=0, epsilon=0.00001, nmax=10_000):
        return newton_raphson(self.to_lambda(), self.derivative().to_lambda(), initial_value, epsilon, nmax)

    def __str__(self):
        if len(self._expressions) == 1:
            return self._expressions[0].__str__()
        accumulator = ""
        for index, expression in enumerate(self._expressions):
            accumulator += '+' if expression.coefficient >= 0 and index > 0 else ""
            accumulator += expression.__str__()
        return accumulator

    def to_dict(self):
        if not self._expressions:
            return {'type': 'Poly', 'data': None}
        return {'type': 'Poly', 'data': [item.to_dict() for item in self._expressions]}

    @staticmethod
    def from_dict(given_dict: dict):
        return Poly([mono.Mono.from_dict(sub_dict) for sub_dict in given_dict['data']])

    @staticmethod
    def from_json(json_content: str):  # Check this method
        parsed_dictionary = json.loads(json_content)
        if parsed_dictionary['type'].strip().lower() != "poly":
            return ValueError(f"Invalid type: {parsed_dictionary['type']}. Expected 'Poly'. ")
        return Poly(mono.Mono.from_dict(mono_dict) for mono_dict in parsed_dictionary['data'])

    @staticmethod
    def import_json(path: str):
        with open(path) as json_file:
            return Poly.from_json(json_file.read())

    def python_syntax(self):
        accumulator = ""
        for index, expression in enumerate(self._expressions):
            accumulator += '+' if expression.coefficient >= 0 and index > 0 else ""
            accumulator += expression.python_syntax()
        return accumulator

    def __fetch_variables_set(self) -> set:
        return {json.dumps(mono_expression.variables_dict) for mono_expression in self._expressions}

    def simplify(
            self):  # TODO: create a new way to simplify polynomials, without re-creating it each time with overhead
        """ simplifying a polynomial"""
        if len(self._expressions) == 0:
            self._expressions = [mono.Mono(0)]
            return
        different_variables: set = self.__fetch_variables_set()
        if "{}" in different_variables:  # TODO: find the source of this stupid bug.
            different_variables.remove("{}")
            different_variables.add("null")
        new_expressions = []
        for variable_dictionary in different_variables:
            if variable_dictionary == 'null':
                same_variables = [expression for expression in self._expressions if
                                  json.dumps(expression.variables_dict) in ("null", "{}")]
            else:
                same_variables = [expression for expression in self._expressions if
                                  json.dumps(expression.variables_dict) == variable_dictionary]

            if len(same_variables) > 1:
                # TODO: BUG ? MONO EXPRESSIONS SOMEHOW GOT MONO COEFFICIENT?
                assert all(
                    isinstance(same_variable.coefficient, (int, float)) for same_variable in
                    same_variables), "Bug detected.."
                coefficients_sum: float = sum(
                    same_variable.coefficient for same_variable in same_variables)
                if coefficients_sum != 0:
                    new_expressions.append(
                        Mono(coefficient=coefficients_sum, variables_dict=same_variables[0].variables_dict))
            elif len(same_variables) == 1:
                if same_variables[0].coefficient != 0:
                    new_expressions.append(same_variables[0])
        self._expressions = new_expressions
        self.sort()

    def sorted_expressions_list(self) -> list:
        """

        :return:
        """
        sorted_exprs = sorted_expressions(
            [expression for expression in self._expressions if expression.variables_dict not in (None, {})])
        free_number = sum(
            expression.coefficient for expression in self._expressions if expression.variables_dict in (None, {}))
        if free_number != 0:  # No reason to add a trailing zero
            sorted_exprs.append(Mono(free_number))
        return sorted_exprs

    def sort(self):
        """
        sorts the polynomial's expression by power, for example : 6 + 3x^2 + 2x  -> 3x^2 + 2x + 6

        :return:
        """
        self._expressions = self.sorted_expressions_list()

    def sorted(self):
        """

        :return:
        """
        return Poly(self.sorted_expressions_list())

    def __len__(self):
        return len(self.expressions)

    def contains_variable(self, variable: str) -> bool:
        """
        Checking whether a certain given variable appears in the expression. For example 'x' does appear in 3x^2 + 5

        :param variable: The variable to be looked for ( type str ). For example : 'x', 'y', etc.
        :return: Returns True if the variable appears in the expression, otherwise False.
        """
        for mono_expression in self._expressions:
            # It only has to appear in at least 1 Mini-IExpression
            if mono_expression.contains_variable(variable):
                return True
        return False

    def __contains__(self, item):
        """
        Determines whether a Poly contains a certain value. for example, 3x^2+5x+7 contains 5x, but doesn't contain 8.
        :param item: allowed types: int,float,str,Mono,Poly
        :return:
        """
        if isinstance(item, (int, float, str)):
            item = Mono(item)

        if isinstance(item, Mono):
            return item in self._expressions
        elif isinstance(item, Poly):
            # Meaning it's smaller and thus can't contain the items
            if len(self.expressions) < len(item._expressions):
                return False
            # if it contains all items, return True
            return all(item in self.expressions for item in item._expressions)
        else:
            raise TypeError(
                f"Poly.__contains__(): unexpected type {type(item)}, expected types: int,float,str,Mono,Poly")

    def __copy__(self):
        return Poly([expression.__copy__() for expression in self._expressions])

    def to_lambda(self):
        """ Returns a lambda expression from the Polynomial"""
        return to_lambda(self.__str__(), self.variables)

    def plot(self, start: float = -10, stop: float = 10,
             step: float = 0.01, ymin: float = -10, ymax: float = 10, text=None, show_axis=True, show=True,
             fig=None, ax=None, formatText=True, values=None):
        lambda_expression = self.to_lambda()
        num_of_variables = self.num_of_variables
        if text is None:
            text = self.__str__()
        if num_of_variables == 0:  # TODO: plot this in a number axis
            raise ValueError("Cannot plot a polynomial with 0 variables_dict")
        elif num_of_variables == 1:
            plot_function(lambda_expression, start=start, stop=stop, step=step, ymin=ymin, ymax=ymax, title=text,
                          show_axis=show_axis, show=show, fig=fig, ax=ax, formatText=formatText, values=values)
        elif num_of_variables == 2:
            # TODO: update the parameters
            plot_function_3d(lambda_expression, start=start,
                             stop=stop, step=step)
        else:
            raise ValueError(
                "Cannot plot a function with more than two variables_dict (As for this version)")

    def to_Function(self):
        return Function(self.__str__())

    def gcd(self):
        """Greatest common divisor of the expressions: for example, for the expression 3x^2 and 6x,
        the result would be 3x"""
        gcd_coefficient = gcd(self.coefficients())
        # If there's a free number
        if any(not expression.variables_dict for expression in self._expressions):
            return Mono(gcd_coefficient)
        gcd_algebraic = Mono(gcd_coefficient)
        my_variables = self.variables
        for variable in my_variables:
            if all(variable in expression.variables_dict for expression in self._expressions):
                powers = [expression.variables_dict[variable]
                          for expression in self._expressions]
                if gcd_algebraic.variables_dict is not None:
                    gcd_algebraic.variables_dict = {
                        **gcd_algebraic.variables_dict, **{variable: min(powers)}}
                else:
                    gcd_algebraic.variables_dict = {variable: min(powers)}
        return gcd_algebraic

    def divide_by_gcd(self):
        return self.__itruediv__(self.gcd())


def poly_frac_from_str(expression: str, get_tuple=False):
    # TODO: implement it better concerning parenthesis and '/' in later versions
    """
    Generates a PolyFraction object from a given string

    :param expression: The given string that represents a polynomial fraction
    :param get_tuple : If set to True, the a tuple of length 2 with the numerator at index 0 and the denoominator at index 1 will be returned.
    :return: Returns a new PolyFraction object, unless get_tuple is True, and then returns the corresponding tuple.
    """
    first_expression, second_expression = expression.split('/')
    if get_tuple:
        return Poly(first_expression), Poly(second_expression)
    return PolyFraction(Poly(first_expression), Poly(second_expression))
