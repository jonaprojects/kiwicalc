def simplify_linear_expression(expression: str, variables: Iterable[str], format_abs=False, format_factorial=False) -> dict:
    if format_abs:
        expression = handle_abs(expression)
    if format_factorial:
        expression = handle_abs(expression)
    expr = expression.replace("-", "+-").replace(" ", "")
    expressions = [num for num in expr.split(
        "+") if num != '' and num is not None]
    if isinstance(variables, dict):
        new_dict = variables.copy()
    else:
        new_dict = {variable_name: 0 for variable_name in variables}
    if "number" not in new_dict:
        new_dict["number"] = 0
    for item in expressions:
        if item[-1].isalpha() or contains_from_list(allowed_characters, item):
            if item[-1] in new_dict.keys():
                if len(item) == 1:
                    item = f"1{item}"
                elif len(item) == 2 and item[0] == '-':
                    item = f"-1{item[-1]}"
                new_dict[item[-1]] += float(item[:-1])
            elif not is_number(item):
                raise ValueError(f"Unrecognized expression {item}")
        else:
            new_dict["number"] += float(item)
    return new_dict
