"""Verification script for notebooks/colab/efficientnetv2b0_exp_a_complete_training.ipynb.
"""

import ast
import json
from pathlib import Path
import nbformat as nbf

def verify_notebook(target_path: Path):
    assert target_path.is_file(), f"Notebook file not found at {target_path}"
    print(f"\n============================================================")
    print(f"VERIFYING NOTEBOOK: {target_path}")
    print(f"============================================================")

    with open(target_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    # 1. Validate nbformat
    nbf.validate(nb)
    print("[OK] nbformat validation passed cleanly.")

    # 2. Check cells and AST parse Python code cells
    code_cells_count = 0
    markdown_cells_count = 0

    print(f"\nTotal Cells: {len(nb.cells)}")
    print("-" * 60)

    cell_titles = []
    for idx, cell in enumerate(nb.cells, start=1):
        ctype = cell.cell_type
        source = cell.source
        first_line = source.split("\n")[0] if source else ""

        safe_line = first_line[:70].encode("ascii", "replace").decode("ascii")
        if ctype == "markdown":
            markdown_cells_count += 1
            print(f"Cell {idx:02d} [Markdown] : {safe_line}")
            if first_line.startswith("#"):
                cell_titles.append(first_line)
        elif ctype == "code":
            code_cells_count += 1
            print(f"Cell {idx:02d} [Code]     : {safe_line}")
            # Strip IPython magic commands (e.g. !pip, !git, %cd) for ast.parse
            python_lines = []
            for line in source.split("\n"):
                if line.strip().startswith("!") or line.strip().startswith("%"):
                    python_lines.append(f"# {line}")
                else:
                    python_lines.append(line)
            python_code = "\n".join(python_lines)

            try:
                ast.parse(python_code)
            except SyntaxError as e:
                raise SyntaxError(f"Syntax error in Cell {idx}: {e}") from e

    print("-" * 60)
    print(f"Markdown cells: {markdown_cells_count}, Code cells: {code_cells_count}")
    print("[OK] All Python code cells passed AST syntax parsing successfully.")

if __name__ == "__main__":
    verify_notebook(Path("notebooks/colab/efficientnetv2b0_exp_a_complete_training.ipynb"))
    verify_notebook(Path("notebooks/colab/inceptionv3_exp_a_complete_training.ipynb"))
