"""
tools.py — Tools the model is allowed to call (the "tool integration" concept).

HOW TOOL CALLING WORKS:
  1. We describe a tool to the model (name, what it does, input schema).
  2. The model decides, mid-answer, that it needs it and replies with a
     `tool_use` block: "call calculate with expression='1299 * 0.85'".
  3. THE MODEL DOES NOT RUN ANYTHING. Our code runs the function.
  4. We send the result back as a `tool_result`, and the model continues.

WHY A CALCULATOR?
  Language models predict text; they are unreliable at arithmetic. If a policy
  says "15% restocking fee on a 4,299 rupee item", we want an exact number from
  real code, not a guess.

SECURITY NOTE (a common interview question):
  The model's input is untrusted text. Using Python's eval() on it would let a
  prompt-injected document run arbitrary code. Instead we parse the expression
  into an AST and only allow plain arithmetic nodes.
"""

import ast
import operator

CALCULATE_TOOL = {
    "name": "calculate",
    "description": (
        "Evaluate a basic arithmetic expression exactly. Use this for any "
        "calculation (totals, percentages, fees, date-free math) instead of "
        "computing in your head. Supports + - * / // % ** and parentheses."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '4299 * 0.15' or '(1200 + 800) / 4'",
            }
        },
        "required": ["expression"],
    },
}

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

MAX_EXPONENT = 100  # stops things like 9**9**9 from freezing the server


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("Exponent too large")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Only plain arithmetic is allowed")


def calculate(expression: str) -> str:
    """Returns the result as a string, or an 'Error: ...' string the model can read and react to."""
    try:
        cleaned = expression.replace(",", "").strip()  # tolerate "4,299"
        tree = ast.parse(cleaned, mode="eval")
        result = _eval_node(tree.body)
        if isinstance(result, float):
            result = round(result, 6)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:  # syntax errors, disallowed nodes, overflow
        return f"Error: {exc}"


def run_tool(name: str, tool_input: dict) -> str:
    """Dispatcher: maps a tool name from the model to the real Python function."""
    if name == "calculate":
        return calculate(tool_input.get("expression", ""))
    return f"Error: unknown tool '{name}'"
