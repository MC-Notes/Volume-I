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
	   	python run_notebook.py $notebook;
	done;
done;
	   