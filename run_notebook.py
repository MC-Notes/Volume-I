#!/usr/bin/env python
# encoding: utf-8

import sys
import os

from optparse import OptionParser

__all__ = []
__version__ = 0.1
__date__ = '2016-11-18'
__updated__ = '2016-11-18'

DEBUG = 1
TESTRUN = 0
PROFILE = 0

def main(argv=None):
    argv = sys.argv[1:]
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    notebook = argv[0]
    note_folder = os.path.dirname(notebook)
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
    print('Running notebook {} ...'.format(argv[0]))
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': note_folder}})

    with open('{}/executed_notebook.ipynb'.format(note_folder), 'wt') as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    sys.exit(main())