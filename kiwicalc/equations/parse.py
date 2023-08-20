class ParseEquation:

    @staticmethod
    def parse_polynomial(equation: str):
        variables = get_equation_variables(equation)
        if len(variables) != 1:
            raise ValueError(
                "can only parse quadratic equations with 1 variable")
        variable = variables[0]
        first_side, second_side = equation.split("=")
        first_dict = ParseExpression.parse_polynomial(
            first_side, variables=variables)
        second_dict = ParseExpression.parse_polynomial(
            second_side, variables=variables)
        add_or_sub_coefficients(
            first_dict[variable], second_dict[variable], copy_first=False, mode='sub')
        return first_dict[variable] + [first_dict['free'] - second_dict['free']]

    @staticmethod
    def parse_quadratic(equation: str, strict_syntax=False):  # TODO: check and fix this
        if strict_syntax:
            return ParseExpression.parse_quadratic(equation, strict_syntax=True)
        return ParseEquation.parse_polynomial(equation)
