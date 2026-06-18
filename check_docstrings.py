import ast
import os

def check_docstrings(path):
    missing = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.py'):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r') as fp:
                        tree = ast.parse(fp.read(), filename=filepath)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            if not ast.get_docstring(node):
                                missing.append((filepath, node.lineno, node.name, type(node).__name__))
                except:
                    pass
    return missing

missing = check_docstrings('ev_qa_framework')
print(f'Total missing docstrings: {len(missing)}')
for f, l, n, t in missing:
    print(f'{f}:{l} {t} {n}')
