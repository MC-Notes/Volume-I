#!/bin/bash
#set -e # halt on error
for folder in $( ls -d */ )
do
    printf "+++++++++++++++++++++++++++++++ \n";
	notebook="$folder$( ls $folder | grep -v $executed_file$ | grep $note_file_ext$ )"

    printf "Processing $folder...\n";
    check_files;
    file_check=$?
    if [ $file_check == 0 ];
    then
        printf "+++++++++++++++++++++++++++++++ \n\n";
        continue; # not running in docs or backend
    fi;
    if [ $file_check == 2 ];
    then
        printf "+++++++++++++++++++++++++++++++ \n\n";
        continue; # files are missing, we keep running for other folders (can be fixed by seperate branches later)
    fi;
    
    if [ ! -f $folder$executed_file ]; # || [ ! -f $folder/executed_notebook.md ]; # Only run if not already:
    then
        # install requirements
        if [ "$CI" == "true" ]; then
            source backend/$backend/$install_requirements $folder $folder$reqs;
        fi;
        #printf "\n";
        echo Running notebook $notebook ...;
        source backend/$backend/$execute_note $notebook $folder$executed_file;
        if [ "$TRAVIS_PULL_REQUEST" == "false" ]; 
        then
            echo "Adding executed notebook to github ...";
            git add $folder$executed_file;
            git add $folder$metadata;
            git commit -m "new: Executed notebook $notebook";
        fi;
    else
        #printf "\n";
        echo $notebook already run, not rerunning.;
    fi;

    if [ "$TRAVIS_PULL_REQUEST" == "false" ]; 
        then
        echo creating docs collection;
        python backend/travis_make_docs.py $folder $folder$executed_file;
        git commit -m "new: Added docs files for notebook $notebook"    
    fi;
    
    if [ ! -f $folder/zenodo_upload.yml ];
    then
        if [ "$TRAVIS_PULL_REQUEST" == "false" ]; 
        then
            printf "\n";
            echo Uploading $folder to zenodo;
            python backend/zenodo_upload_doi.py $ZENODO_ACCESS $ZENODO_ACCESS_TOKEN $folder$metadata $folder$executed_file $folder$reqs
            git add ${folder}zenodo_upload.yml;
            git commit -m "new: $SHA Uploaded to zenodo $folder";
        fi;
    else
        #printf "\n";
        printf "$folder already uploaded\n"; # as \n\n$( cat $folder/zenodo_upload.yml )\n";
    fi;
    printf "+++++++++++++++++++++++++++++++ \n\n";
done;

if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
    if [ "$CI" == "true" ]; then
        #if [ -z `git diff --exit-code` ]; then
        #    echo "No changes to the output on this push; exiting."
        #    exit 0
        #fi
        git push $SSH_REPO $TRAVIS_BRANCH;
    else
        echo Updated tree, see git status for details.;
    fi;
fi

source deactivate