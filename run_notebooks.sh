#!/bin/bash

for folder in $( ls -d */ )
do
    echo $folder
    for notebook in $( ls $folder/*.ipynb )
    do
        if [ -a $folder/requirements.txt ]
        then
            pip install -r $folder/requirements.txt;
        fi;
        if [ ! -f $folder/executed_notebook.ipynb ] # Only run if not already:
        then
            echo Running notebook $notebook...;
            python run_notebook.py $notebook;
            echo Adding exectued notebook to github...;
            git add $folder/executed_notebook.ipynb;
            git commit -m "new: Executed notebook $notebook";
        else
            echo Notebook $notebook already run, not rerunning.;
        fi;
    done;
done;
	   