class ParseExpression:

    @staticmethod
    def parse_linear(expression, variables):
        pass

    @staticmethod
    def unparse_linear(variables_dict: dict, free_number: float):
        accumulator = []
        for variable, coefficients in variables_dict.items():
            for coefficient in coefficients:
                if coefficient == 0:
                    continue
                coefficient_str = format_coefficient(coefficient)
                if coefficient > 0:
                    accumulator.append(f"+{coefficient_str}{variable}")
                else:
                    accumulator.append(f"{coefficient_str}{variable}")
        accumulator.append(format_free_number(free_number))
        result = "".join(accumulator)
        if not result:
            return "0"
        if result[0] == '+':
            return result[1:]
        return result

    @staticmethod
    def parse_quadratic(expression: str, variables=None, strict_syntax=True):
        expression = expression.replace(" ", "")
        if variables is None:
            variables = get_equation_variables(expression)
        if strict_syntax:
            if len(variables) != 1:
                raise ValueError(f"Strict quadratic syntax must contain exactly 1 variable, found {len(variables)}")
            variable = variables[0]
            if '**' in expression:
                expression = expression.replace('**', '^')
            a_expression_index: int = expression.find(f'{variable}^2')
            if a_expression_index == -1:
                raise ValueError(f"Didn't find expression containing '{variable}^2' ")
            elif a_expression_index == 0:
                a = 1
            else:
                a = extract_coefficient(expression[:a_expression_index])
            b_expression_index = expression.rfind(variable)
            if b_expression_index == -1:
                b = 0
                c_str = expression[a_expression_index + 3:]
            else:
                b = extract_coefficient(expression[a_expression_index + 3:b_expression_index])
                c_str = expression[b_expression_index + 1:]
            if c_str == '':
                c = 0
            else:
                c = extract_coefficient(c_str)
            return {variable: [a, b], 'free': c}
        else:
            return ParseExpression.parse_polynomial(expression, variables, strict_syntax)

    @staticmethod
    def parse_cubic(expression: str, variables, strict_syntax=True):
        expression = expression.replace(" ", "")
        if strict_syntax:
            print("reached here for cubic")
            if len(variables) != 1:
                raise ValueError(f"Strict cubic syntax must contain exactly 1 variable, found {len(variables)}")
            variable = variables[0]
            if '**' in expression:
                expression = expression.replace('**', '^')
            a_expression_index: int = expression.find(f'{variable}^3')
            if a_expression_index == -1:
                raise ValueError(f"Didn't find expression containing '{variable}^3' ")
            elif a_expression_index == 0:
                a = 1
            else:
                a = extract_coefficient(expression[:a_expression_index])
            b_expression_index = expression.find(f'{variable}^2')
            if b_expression_index == -1:
                b_expression_index = a_expression_index
                b = 0
            else:
                b = extract_coefficient(expression[a_expression_index + 3:b_expression_index])

            c_expression_index = expression.rfind(variable)
            if c_expression_index == -1:
                c = 0
                c_expression_index = b_expression_index
            else:
                c = extract_coefficient(expression[b_expression_index + 3:c_expression_index])
            d_str = expression[c_expression_index + 1:]
            if d_str == '':
                d = 0
            else:
                d = extract_coefficient(d_str)
            return {variable: [a, b, c], 'free': d}
        else:
            return ParseExpression.parse_polynomial(expression, variables, strict_syntax)

    @staticmethod
    def parse_quartic(expression: str, variables, strict_syntax=True):
        expression = expression.replace(" ", "")
        if strict_syntax:
            if len(variables) != 1:
                raise ValueError(f"Strict quadratic syntax must contain exactly 1 variable, found {len(variables)}")
            variable = variables[0]
            if '**' in expression:
                expression = expression.replace('**', '^')
            a_expression_index: int = expression.find(f'{variable}^4')
            if a_expression_index == -1:
                raise ValueError(f"Didn't find expression containing '{variable}^4' ")
            elif a_expression_index == 0:
                a = 1
            else:
                a = extract_coefficient(expression[:a_expression_index])
            b_expression_index = expression.find(f'{variable}^3')
            if b_expression_index == -1:
                b_expression_index = a_expression_index
                b = 0
            else:
                b = extract_coefficient(expression[a_expression_index + 3:b_expression_index])

            c_expression_index = expression.find(f'{variable}^2')
            if c_expression_index == -1:
                c = 0
                c_expression_index = b_expression_index
            else:
                c = extract_coefficient(expression[b_expression_index + 3:c_expression_index])

            d_expression_index = expression.rfind(variable)
            if d_expression_index == -1:
                d = 0
                d_expression_index = c_expression_index
            else:
                d = extract_coefficient(expression[c_expression_index + 3:d_expression_index])

            e_str = expression[d_expression_index + 1:]
            if e_str == '':
                e = 0
            else:
                e = extract_coefficient(e_str)

            return {variable: [a, b, c, d], 'free': e}
        else:
            return ParseExpression.parse_polynomial(expression, variables)

    @staticmethod
    def parse_polynomial(expression: str, variables=None, strict_syntax=True, numpy_array=False, get_variables=False):
        if variables is None:
            variables = list({character for character in expression if character.isalpha()})
        expression = clean_spaces
        mono_expressions = split_expression(expression)
        if numpy_array:
            variables_dict = {variable: np.array([], dtype='float64') for variable in variables}
        else:
            variables_dict = {variable: [] for variable in variables}
        variables_dict['free'] = 0
        for mono in mono_expressions:
            coefficient, variable, power = ParseExpression._parse_monomial(mono, variables)
            if power == 0:
                variables_dict['free'] += coefficient
            else:
                coefficient_list = variables_dict[variable]
                if power > len(coefficient_list):
                    zeros_to_add = int(power) - len(coefficient_list) - 1
                    if numpy_array:
                        coefficient_list = np.pad(coefficient_list, (zeros_to_add, 0), 'constant', constant_values=(0,))
                        variables_dict[variable] = np.insert(coefficient_list, 0, coefficient)
                    else:
                        for _ in range(zeros_to_add):
                            coefficient_list.insert(0, 0)
                        coefficient_list.insert(0, coefficient)
                else:
                    coefficient_list[len(coefficient_list) - int(power)] += coefficient
        if not get_variables:
            return variables_dict
        return variables_dict, variables

    @staticmethod
    def unparse_polynomial(parsed_dict: dict, syntax=""):
        """Taking a parsed polynomial and returning a string from it"""
        accumulator = []
        if syntax not in ("", "pythonic"):
            warnings.warn(f"Unrecognized syntax: {syntax}. Either use the default or 'pythonic' ")
        for variable, coefficients in parsed_dict.items():
            if variable == 'free':
                continue
            sub_accumulator, num_of_coefficients = [], len(coefficients)
            for index, coefficient in enumerate(coefficients):
                if coefficient != 0:
                    coefficient_str = format_coefficient(round_decimal(coefficient))
                    if coefficient_str not in ('', '-') and syntax == 'pythonic':
                        coefficient_str += '*'
                    power = len(coefficients) - index
                    sign = '' if coefficient < 0 or (not accumulator and not sub_accumulator) else '+'
                    if power == 1:
                        sub_accumulator.append(f"{sign}{coefficient_str}{variable}")
                    else:
                        if syntax == 'pythonic':
                            sub_accumulator.append(f"{sign}{coefficient_str}{variable}**{power}")
                        else:
                            sub_accumulator.append(f"{sign}{coefficient_str}{variable}^{power}")
            accumulator.extend(sub_accumulator)
        free_number = parsed_dict['free']
        if free_number != 0 or not accumulator:
            sign = '' if free_number < 0 or not accumulator else '+'
            accumulator.append(f"{sign}{round_decimal(free_number)}")
        return "".join(accumulator)

    @staticmethod
    def _parse_monomial(expression: str, variables):
        """ Extracting the coefficient an power from a monomial, this method is used while parsing polynomials"""
        # Check which variable appears in the expression
        print(expression)
        variable_index = -1
        for suspect_variable in variables:
            suspect_variable_index = expression.find(suspect_variable)
            if suspect_variable_index != -1:
                variable_index = suspect_variable_index
                break
        if variable_index == -1:
            # If we haven't found any variable, that means that the expression is a free number
            try:
                return float(expression), 'free', 0
            except ValueError:
                raise ValueError("Couldn't parse the expression! Found no variables, but the free number isn't valid.")
        else:
            variable = expression[variable_index]
            try:
                coefficient = extract_coefficient(expression[:variable_index])
            except ValueError:
                raise ValueError(f"Encountered an invalid coefficient '{expression[:variable_index]}' while"
                                 f"parsing the monomial '{expression}'")
            power_index = expression.find('^')
            if power_index == -1:
                return coefficient, variable, 1
            try:
                power = float(expression[power_index + 1:])
                return coefficient, variable, power
            except ValueError:
                raise ValueError(f"encountered an invalid power '{expression[power_index + 1:]} while parsing the"
                                 f"monomial '{expression}'")

    @staticmethod
    def to_coefficients(expression: str, variable=None, strict_syntax=True, get_variable=False):
        expression = clean_spacesession)
        if variable is None:
            variables = get_equation_variables(expression)
            num_of_variables = len(variables)
            if num_of_variables == 0:
                return [float(expression)]
            elif num_of_variables != 1:
                raise ValueError(f"Can only parse polynomials with 1 variable, but got {num_of_variables}")
            variable = variables[0]
        mono_expressions = split_expression(expression)
        coefficients_list = [0]
        for mono in mono_expressions:
            coefficient, variable, power = ParseExpression._parse_monomial(mono, (variable,), strict_syntax)
            if power == 0:
                coefficients_list[-1] += coefficient
            else:
                if power > len(coefficients_list) - 1:
                    zeros_to_add = int(power) - len(coefficients_list)
                    for _ in range(zeros_to_add):
                        coefficients_list.insert(0, 0)
                    coefficients_list.insert(0, coefficient)
                else:
                    coefficients_list[len(coefficients_list) - int(power) - 1] += coefficient
        if not get_variable:
            return coefficients_list
        return coefficients_list, variable

    @staticmethod
    def coefficients_to_str(coefficients, variable='x', syntax=""):
        """Taking a parsed polynomial and returning a string from it"""
        accumulator = []
        if syntax not in ("", "pythonic"):
            warnings.warn(f"Unrecognized syntax: {syntax}. Either use the default or 'pythonic' ")
        num_of_coefficients = len(coefficients)
        if num_of_coefficients == 0:
            raise ValueError("At least 1 coefficient is required")
        elif num_of_coefficients == 1:
            return f"{coefficients[0]}"
        for index in range(num_of_coefficients - 1):
            coefficient = coefficients[index]
            if coefficient != 0:
                coefficient_str = format_coefficient(round_decimal(coefficient))
                if coefficient_str not in ('', '-') and syntax == 'pythonic':
                    coefficient_str += '*'
                power = len(coefficients) - index - 1
                sign = '' if coefficient < 0 or not accumulator else '+'
                if power == 1:
                    accumulator.append(f"{sign}{coefficient_str}{variable}")
                else:
                    if syntax == 'pythonic':
                        accumulator.append(f"{sign}{coefficient_str}{variable}**{power}")
                    else:
                        accumulator.append(f"{sign}{coefficient_str}{variable}^{power}")
        free_number = coefficients[-1]
        if free_number != 0 or not accumulator:
            sign = '' if free_number < 0 or not accumulator else '+'
            accumulator.append(f"{sign}{round_decimal(free_number)}")
        return "".join(accumulator)
