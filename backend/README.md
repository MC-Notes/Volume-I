# Backend repository
Tags represent a workflow versions. This repository holds the different version of the backend to be used by the volumes. The volumes define the backend version to use in their `mc_notes.yml` file. 

## Environment variables
Environment variables for backends are available in every script. We source scripts from the main thread to retain environment variables. Variables are:

| variable | description |
|----------|-------------|
| $backend | the backend set in the environment config.yml |
| $workflow_version | the version of workflow to use |
| $notebook | the current notebook being run |
| $note_file_ext | the file extension for notebooks |
| $executed_file | filename for the note after execution |
| $metadata | filename for the metadata yaml file |
| $reqs | filename for requirement files |
| $folder | current notebook folder running |


## Adding your own environment

To add a new environment, we need to define how to install requirements and how to run the notes submitted. Here we describe the files in the python environment as an example of the process of creating your own backend for another language.

### `config.yml`
Defines the variables for the build process:

```yaml
# File to store metadata about publication (name, affiliation, etc.)
metadata: metadata.yml
# File storing requirements for publication note (pandas, seaborn etc for python notebooks)
reqs: requirements.txt
# File name for executed file in publication. This file will be used for DOI .upload 
executed_file: executed_notebook.ipynb
# Extension for note publication notebook
note_file_ext: .ipynb
# Script to execute. This command will be run with 2 arguments: $<path/to/publication/note> $<path/to/publication/executed_file>
execute_note: run_note.sh
# command line to install requirements of publications. This command will be run with arguments: $<path/to/publication $<path/to/publication/requirements.txt>
install_requirements: install_reqs.sh
```

### `run_note.sh`
The bash/shell script to run a note. In python this calls the a python script, `run_notebook.py` with two arguments. `run_note.sh` gets two arguments, the notebook to run (relative path from the root of the repository) and the output file (`executed_notebook.ipynb` for python). The name of the executed notebook can be set in the config above. Note, that all variables can be accessed in this script. So in python the script is simple and just calls the underlying python routine:

```bash
#!/bin/bash
python backend/$backend/run_notebook.py $1 $2
```

### `install_reqs.sh`
The script to install requirements, with two arguments the folder to create the environment for and the requirements file as defined in the `config.yml` (see above). Here we create a new environment for each note and install the requirements to prevent collisions, or not to miss any missing requirements:

```bash
#!/bin/bash
env_name=${1%/}
conda create --force --name $env_name --clone base;
source activate $env_name
pip install -r $2;
```
