def handle_trigo_calculation(expression: str):
    """ getting the result of a single trigonometric operation, e.g : sin(90) -> 1"""
    selected_operation = [
        op for op in TRIGONOMETRY_CONSTANTS.keys() if op in expression]
    selected_operation = selected_operation[0]
    start_index = expression.find(
        selected_operation) + len(selected_operation) + 1
    coef = expression[:expression.find(selected_operation)]
    if coef == "" or coef is None or coef == "+":
        coef = 1
    elif coef == '-':
        coef = 1
    else:
        coef = float(coef)
    parameter = expression[start_index] if expression[start_index].isdigit() or expression[
        start_index] == '-' else ""
    for i in range(start_index + 1, expression.rfind(')')):
        parameter += expression[i]
    if is_evaluatable(parameter):
        parameter = float(eval(parameter))
    parameter = -float(parameter) if expression[0] == '-' else float(parameter)
    return round_decimal(coef * TRIGONOMETRY_CONSTANTS[selected_operation](parameter))


def handle_trigo_expression(expression: str):
    """ handles a whole trigonometric expression, for example: 2sin(90)+3sin(60)"""
    expressions = split_expression(expression)
    result = 0
    for expr in expressions:
        result += handle_trigo_calculation(expr)
    return result
