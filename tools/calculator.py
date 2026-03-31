"""AST-based safe calculator — no raw eval(), no code execution."""
from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool

# Whitelist of allowed AST node types → operator functions
_SAFE_OPS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Non-numeric constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(operand)

    raise ValueError(f"Disallowed expression: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Supports: +, -, *, /, //, %, ** and parentheses.
    Does NOT support: function calls, variables, imports, or any code execution.

    Args:
        expression: A math expression string, e.g. '(3 + 4) * 2 / 7'

    Returns:
        The numeric result as a string, or an error message.

    Examples:
        calculator("2 + 2")           -> "4.0"
        calculator("(10 ** 2) / 4")   -> "25.0"
        calculator("17 % 5")          -> "2.0"
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        # Return clean integer string when result has no fractional part
        if result == int(result):
            return str(int(result))
        return str(round(result, 10))
    except ZeroDivisionError:
        return "Error: Division by zero."
    except SyntaxError as exc:
        return f"Error: Invalid expression syntax — {exc}"
    except ValueError as exc:
        return f"Error: {exc}"
