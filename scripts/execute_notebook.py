"""Execute every cell in a fresh Python process, preserving real rich outputs.

The default IPython runner needs no network sockets. --kernel uses a normal
Jupyter kernel when the host supports local sockets. Both stop on cell errors.
"""
import argparse
from pathlib import Path
import sys

import nbformat
ROOT = Path(__file__).resolve().parents[1]
notebook_path = ROOT / "notebooks/financial_fraud_detection_final.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kernel", action="store_true", help="Use a networked Jupyter kernel instead of fresh-process IPython.")
args = parser.parse_args()
if args.kernel:
    from nbclient import NotebookClient
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    backend = "Jupyter kernel"
else:
    import os
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    os.chdir(ROOT)
    shell = InteractiveShell.instance()
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        cell.outputs = []
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f"Notebook code cell {count} failed: {captured.stderr}\n{captured.stdout}") from (result.error_before_exec or result.error_in_exec)
        cell.execution_count = count
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
        print(f"Executed code cell {count}", flush=True)
    backend = "Fresh-process IPython; no networked kernel"
notebook.metadata["execution_backend"] = backend
nbformat.write(notebook, notebook_path)
errors = [output for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []) if output.output_type == "error"]
print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells; {len(errors)} errors.")
sys.exit(bool(errors))
