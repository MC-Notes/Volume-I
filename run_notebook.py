#!/usr/bin/env python
# encoding: utf-8

import sys
import os

def main(argv=None):
    argv = sys.argv[1:]
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbconvert.preprocessors.execute import CellExecutionError
    from nbconvert.exporters import MarkdownExporter

    notebook = argv[0]
    note_folder = os.path.dirname(notebook)
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
    
    #print('Running notebook {} ...'.format(argv[0]))
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        nb, resources = ep.preprocess(nb, {'metadata': {'path': note_folder}})
    except CellExecutionError as c:
        print(c)
        return 2

    with open('{}/executed_notebook.ipynb'.format(note_folder), 'wt') as f:
        nbformat.write(nb, f)

    with open('{}/executed_notebook.md'.format(note_folder), 'wt') as f:
        mdconvert = MarkdownExporter()
        mdnb, mdresources = mdconvert.from_notebook_node(nb, resources)
        f.write(mdnb)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
