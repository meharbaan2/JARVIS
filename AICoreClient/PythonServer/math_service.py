import ast
import operator
import re


class MathService:
    """Deterministic math helper with optional SymPy support."""

    def __init__(self):
        self.sympy = None
        try:
            import sympy
            self.sympy = sympy
        except Exception:
            self.sympy = None

    def health(self):
        return {
            "available": True,
            "symbolic": self.sympy is not None,
            "provider": "sympy" if self.sympy else "safe_arithmetic",
        }

    def handle(self, text):
        query = (text or "").strip()
        if not query:
            return "I need a math expression or equation to solve."

        if self.sympy:
            result = self._handle_with_sympy(query)
            if result:
                return result

        arithmetic = self._handle_arithmetic(query)
        if arithmetic:
            return arithmetic

        return (
            "I could not parse that as a deterministic math request. "
            "Try formats like `calculate 2*(3+4)`, `solve x^2 - 4 = 0`, "
            "`derivative of x^3`, or `integrate x^2`."
        )

    def _handle_with_sympy(self, query):
        sympy = self.sympy
        lowered = query.lower()
        expression_text = _extract_expression(query)
        x = sympy.symbols("x")

        try:
            if "derivative" in lowered or "differentiate" in lowered:
                expr = self._sympify(_after_keyword(query, ["derivative of", "differentiate", "derivative"]) or expression_text)
                result = sympy.diff(expr, x)
                return f"Derivative with respect to x: {sympy.sstr(result)}"

            if "integral" in lowered or "integrate" in lowered:
                expr = self._sympify(_after_keyword(query, ["integral of", "integrate", "integral"]) or expression_text)
                result = sympy.integrate(expr, x)
                return f"Indefinite integral with respect to x: {sympy.sstr(result)} + C"

            if "factor" in lowered:
                expr = self._sympify(_after_keyword(query, ["factor"]) or expression_text)
                return f"Factored form: {sympy.sstr(sympy.factor(expr))}"

            if "expand" in lowered:
                expr = self._sympify(_after_keyword(query, ["expand"]) or expression_text)
                return f"Expanded form: {sympy.sstr(sympy.expand(expr))}"

            if "simplify" in lowered:
                expr = self._sympify(_after_keyword(query, ["simplify"]) or expression_text)
                return f"Simplified form: {sympy.sstr(sympy.simplify(expr))}"

            if "=" in expression_text and any(word in lowered for word in ["solve", "equation", "x="]):
                left, right = expression_text.split("=", 1)
                equation = sympy.Eq(self._sympify(left), self._sympify(right))
                solutions = sympy.solve(equation, x)
                return f"Solutions for x: {sympy.sstr(solutions)}"

            if any(word in lowered for word in ["calculate", "evaluate", "solve"]) or _looks_like_expression(expression_text):
                expr = self._sympify(expression_text)
                simplified = sympy.simplify(expr)
                if not getattr(simplified, "free_symbols", set()):
                    if getattr(simplified, "is_Integer", False) or getattr(simplified, "is_Rational", False):
                        return f"Result: {sympy.sstr(simplified)}"
                numeric = simplified.evalf()
                if simplified == numeric:
                    return f"Result: {sympy.sstr(simplified)}"
                return f"Exact result: {sympy.sstr(simplified)}\nApproximation: {numeric}"
        except Exception as exc:
            return f"I could not solve that deterministically: {exc}"

        return None

    def _sympify(self, expression):
        cleaned = _clean_expression(expression)
        try:
            from sympy.parsing.sympy_parser import (
                implicit_multiplication_application,
                standard_transformations,
                parse_expr,
            )
            transformations = standard_transformations + (implicit_multiplication_application,)
            x = self.sympy.symbols("x")
            return parse_expr(cleaned, transformations=transformations, local_dict={"x": x})
        except Exception:
            return self.sympy.sympify(cleaned)

    def _handle_arithmetic(self, query):
        expression = _extract_expression(query)
        if not _looks_like_expression(expression):
            return None
        try:
            result = _safe_eval(expression)
            return f"Result: {result}"
        except Exception:
            return None


def _after_keyword(text, keywords):
    lowered = text.lower()
    for keyword in keywords:
        index = lowered.find(keyword)
        if index >= 0:
            return text[index + len(keyword):].strip(" :")
    return ""


def _extract_expression(text):
    candidate = text.strip()
    candidate = re.sub(
        r"^(please\s+)?(calculate|evaluate|solve|what is|what's|simplify)\s+",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"\?$", "", candidate).strip()
    return candidate


def _clean_expression(expression):
    cleaned = expression.strip()
    cleaned = cleaned.replace("^", "**")
    cleaned = re.sub(r"\bpi\b", "pi", cleaned, flags=re.IGNORECASE)
    return cleaned


def _looks_like_expression(expression):
    return bool(re.search(r"[0-9xX][0-9xX\s+\-*/().=^]*", expression))


_OPERATORS = {
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


def _safe_eval(expression):
    cleaned = _clean_expression(expression)
    tree = ast.parse(cleaned, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")
