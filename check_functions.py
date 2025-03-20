import ast
import sys

def is_function_implemented(func_def):
    """
    Determines if a function is implemented.
    A function is considered "Not Implemented" if its body is empty,
    or if it only contains a single 'pass' statement (after removing an optional docstring).
    """
    body = func_def.body
    # Remove leading docstring if present
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, (ast.Str, ast.Constant)):
        body = body[1:]
    # Check if function body is empty or contains only a pass statement
    if not body:
        return False
    if len(body) == 1 and isinstance(body[0], ast.Pass):
        return False
    return True

def main(filepath):
    try:
        with open(filepath, "r") as f:
            source = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
        
    try:
        tree = ast.parse(source, filename=filepath)
    except Exception as e:
        print(f"Error parsing file: {e}")
        sys.exit(1)
        
    # Iterate over all function definitions in the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            status = "Implemented" if is_function_implemented(node) else "Not Implemented"
            print(f"Function '{node.name}': {status}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_functions.py <path_to_file>")
        sys.exit(1)
    main(sys.argv[1])
