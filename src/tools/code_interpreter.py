"""Safe sandboxed Python Code Interpreter for deterministic mathematical computations in GATE CS."""

import ast
import contextlib
import io
import math
import sys
import time
from typing import Any, Dict


class SecurityError(Exception):
    """Raised when disallowed or unsafe code is detected."""
    pass


class SafeCodeVisitor(ast.NodeVisitor):
    """Inspects Python AST to reject unsafe modules, system calls, and file I/O."""

    DISALLOWED_MODULES = {
        "os", "sys", "subprocess", "shutil", "socket", "http", "urllib",
        "requests", "threading", "multiprocessing", "ctypes", "builtins",
        "importlib", "pickle", "pathlib", "webbrowser",
    }

    DISALLOWED_CALLS = {
        "exec", "eval", "compile", "open", "__import__", "globals",
        "locals", "vars", "dir", "input", "exit", "quit"
    }

    def visit_Import(self, node):
        for alias in node.names:
            base_pkg = alias.name.split(".")[0]
            if base_pkg in self.DISALLOWED_MODULES:
                raise SecurityError(f"Import of module '{alias.name}' is prohibited for safety.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            base_pkg = node.module.split(".")[0]
            if base_pkg in self.DISALLOWED_MODULES:
                raise SecurityError(f"Import from module '{node.module}' is prohibited for safety.")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DISALLOWED_CALLS:
                raise SecurityError(f"Call to '{node.func.id}' is prohibited for safety.")
        self.generic_visit(node)


class CodeInterpreter:
    """Executes safe Python code blocks for mathematical calculations and algorithm simulation."""

    ALLOWED_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "chr": chr,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "math": math,
    }

    def __init__(self, timeout_seconds: float = 3.0):
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str) -> Dict[str, Any]:
        """Executes a Python snippet safely and captures standard output."""
        start_time = time.perf_counter()
        
        # 1. Parse and validate AST
        try:
            tree = ast.parse(code)
            validator = SafeCodeVisitor()
            validator.visit(tree)
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"SyntaxError: {e}",
                "output": "",
                "elapsed_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }
        except SecurityError as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "elapsed_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        # 2. Setup execution context
        safe_globals = {
            "__builtins__": self.ALLOWED_BUILTINS,
            "math": math,
        }
        safe_locals: Dict[str, Any] = {}
        stdout_buf = io.StringIO()

        # 3. Execute with stdout capture
        try:
            with contextlib.redirect_stdout(stdout_buf):
                # Compile to bytecode
                compiled = compile(tree, filename="<calypso_sandbox>", mode="exec")
                exec(compiled, safe_globals, safe_locals)
            
            output = stdout_buf.getvalue().strip()
            
            # If no print statement was used but the last statement was an expression, capture its result
            if not output and tree.body and isinstance(tree.body[-1], ast.Expr):
                # Try evaluating the last expression
                try:
                    last_expr = ast.Expression(body=tree.body[-1].value)
                    last_compiled = compile(last_expr, filename="<calypso_sandbox>", mode="eval")
                    res = eval(last_compiled, safe_globals, safe_locals)
                    output = str(res)
                except Exception:
                    pass

            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "success": True,
                "output": output,
                "error": None,
                "variables": {k: str(v) for k, v in safe_locals.items() if not k.startswith("_")},
                "elapsed_ms": elapsed,
            }

        except Exception as e:
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "success": False,
                "output": stdout_buf.getvalue().strip(),
                "error": f"{type(e).__name__}: {e}",
                "elapsed_ms": elapsed,
            }
