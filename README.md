# MOckup
Thank you for visiting, for now nothing is working yet!

[![Build Status](https://travis-ci.org/MC-Notes/Issue1.svg)](https://travis-ci.org/MC-Notes/Issue1)

This issue of mc-notes includes applications of machine learning techniques to single cell gene expression experiments. 
All experiments and data must be available for free and downloaded by the script automatically.

## How to submit
Submissions are pull requests to the master branch. The review process will be completely open to follow and interaction with the author is greatly appreciated. The work included in notebooks does not need to be published otherwise. This is meant to be a publication platform for usefull scripts for fellow researchers to use in their workflow.

The structure of the notebook should include:
 - Name of the notebook.
 - Author names, contact and short summary (abstract).
 - Cell description and cells to run the code. 
 - Conclusion \& [References](#References)

### Files to include
 - `\<notebook_name\>.ipynb` 
   Jupyter notebook runnable in python 3. 
   The notebook has to run through without any user input. 
   Make sure the notebook runs by itself all the way through without user interaction, as otherwise the submission will not be accepted.
 - `requirements.txt`
   One line per required package for installation. Packages must be provided on pip and installable by pip install \<packagname\>.
 - No other files are allowed. Supporting scripts should be directly embedded into the notebook.

### References
We recommend using calysto document tools from the calysto notebook extensions https://github.com/Calysto/notebook-extensions, which allows you to reference bibtex style and include the bibliography directly into the notebook. 
