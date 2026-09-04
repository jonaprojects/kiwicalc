"""Execute educator notebook cells in IPython without a ZeroMQ kernel.

Requires notebook development tools (nbformat/IPython), not library runtime
dependencies. Uses the same magics and display calls as the notebook. Run from
the repository root with a Python environment containing editable KiwiCalc.
"""
import base64
from pathlib import Path

import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output


def main():
    path = Path('examples/numerical_methods_for_educators.ipynb')
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    shell = TerminalInteractiveShell.instance()
    shell.display_formatter.active_types = ['text/plain', 'text/html', 'image/png']
    code_cells, figures, animations = 0, 0, 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        with capture_output() as captured:
            result = shell.run_cell(cell.source)
        result.raise_error()
        code_cells += 1
        for output in captured.outputs:
            if 'image/png' in output.data:
                figures += 1
                if figures == 1:
                    png = output.data['image/png']
                    path.with_name('numerical_methods_preview.png').write_bytes(base64.b64decode(png))
            if 'text/html' in output.data and 'animation' in output.data['text/html']:
                animations += 1
    print(f'Executed {code_cells} code cells; rendered {figures} figures and {animations} HTML animations')
    assert code_cells >= 10 and figures >= 6 and animations == 2


if __name__ == '__main__':
    main()
