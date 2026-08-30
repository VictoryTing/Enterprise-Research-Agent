from typing import Union


def calculate(expression: str) -> Union[int, float]:
    """
    Calculate a mathematical expression.

    Example:
        calculate("12 * 5")
        -> 60
    """

    try:
        # Basic calculator for arithmetic expressions.
        allowed_chars = "0123456789+-*/(). "

        if not all(char in allowed_chars for char in expression):
            raise ValueError("Expression contains unsupported characters.")

        result = eval(expression, {"__builtins__": {}}, {})

        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not return a number.")

        return result

    except Exception as e:
        raise ValueError(f"Invalid expression: {expression}") from e