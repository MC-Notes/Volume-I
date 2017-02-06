---
layout: page
title: How to submit
subtitle: a new Notebook
---

Submissions are [pull requests](https://help.github.com/articles/about-pull-requests/) from a [fork](https://guides.github.com/activities/forking/) to the master branch of the selected issue. The review process will be completely open to follow and interaction with the author is greatly appreciated. The work included in notebooks does not need to be published otherwise. This is meant to be a publication platform for usefull scripts for fellow researchers to use in their workflow.

The structure of the notebook should include:
 - Name of the notebook.
 - Author names, contact and short summary (abstract).
 - Cell description and cells to run the code. 
 - Conclusion \& [References](#References)

# Files to include
- `\<notebook_name\>.ipynb` 
  Jupyter notebook runnable in python 3. 
  The notebook has to run through without any user input. 
  Make sure the notebook runs by itself all the way through without user interaction, as otherwise the submission will not be accepted.
  **The filename `executed_notebook.ipynb` is reserved and cannot be used as a file name!**
- `requirements.txt`
  One line per required package for installation. Packages must be provided on pip and installable by pip install \<packagname\>.
- `metadata.yml`
  Yaml file containing the metadata for your publication. This file will be used to add a DOI to your notebook for citation. An example file is shown below. Optional entries are marked as optional.
   
  ```yaml
  title: Title of Notebook
  creators:
    - {'name': "last, first", 'affiliation': 'university of ...'}
    - {'name': "last, first", 'affiliation': 'university of ...'}
    ...
  description: "Short description (abstract) of the notebook."
  keywords: ['keyword1', 'keyword2', ... ] # Optional
  contributors: # Optional
    - {'name': "last, first", 'type': 'DataCollector', 'affiliation': 'university of ...'}
    - {'name': "last, first", 'type': 'ContactPerson', 'affiliation': 'university of ...'}
  grants: # Optional
    - 'id':'FP7 grant id'
  ```
  Have a look at https://sandbox.zenodo.org/dev#restapi-rep-meta for a detailed description of the metadata file and options.

*No other files are allowed, accompanying scripts should be included into the notebook at the appropriate position.*
  
# References
We recommend using calysto document tools from the calysto notebook extensions https://github.com/Calysto/notebook-extensions, which allows you to reference bibtex style and include the bibliography directly into the notebook. 
