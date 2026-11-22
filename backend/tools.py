import ast
import operator

CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate a basic arithmetic expression exactly. Use this for any "
            "calculation (totals, percentages, fees, date-free math) instead of "
            "computing in your head. Supports + - * / // % ** and parentheses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '4299 * 0.15' or '(1200 + 800) / 4'",
                }
            },
            "required": ["expression"],
        },
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

MAX_EXPONENT = 100


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
    try:
        cleaned = expression.replace(",", "").strip() 
        tree = ast.parse(cleaned, mode="eval")
        result = _eval_node(tree.body)
        if isinstance(result, float):
            result = round(result, 6)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:  
        return f"Error: {exc}"


def run_tool(name: str, tool_input: dict) -> str:
    if name == "calculate":
        return calculate(tool_input.get("expression", ""))
    return f"Error: unknown tool '{name}'"
