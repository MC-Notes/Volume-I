#!/bin/bash

if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
    REPO=`git config remote.origin.url`;
    SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:};
    SHA=`git rev-parse --verify HEAD`;
    ENCRYPTION_LABEL="$encrypted_a3a89bfc08a4";
    COMMIT_AUTHOR_EMAIL="ibinbei@gmail.com";
    openssl aes-256-cbc -K $encrypted_a3a89bfc08a4_key -iv $encrypted_a3a89bfc08a4_iv -in secrets.tar.enc -out secrets.tar -d
    tar xvf secrets.tar;
    ZENODO_ACCESS_TOKEN=`cat zenodo-access`;
    ZENODO_ACCESS=https://sandbox.zenodo.org/api/deposit/depositions;
    chmod 600 github_deploy;
    chmod 600 zenodo_access;
    eval `ssh-agent -s`;
    ssh-add github_deploy;
    git clone $REPO out
    cd out
    git config user.name "Travis CI";
    git config user.email "$COMMIT_AUTHOR_EMAIL";
fi;

function check_files {
    folder=$1
    reqs=$folder/requirements.txt;
    metadata=$folder/metadata.yml;
    notebook="$folder/$( ls $folder | grep -v 'executed_notebook.ipynb$' | grep '.ipynb$' )"
    local exit_after=0
    
    # Make sure files exist
    test ! -f $reqs && (printf "Missing requirements.txt for $folder, please provide requirements as described in the readme (or empty file if no requirements)." 1>&2; exit_after=1);
    test ! -f $metadata && (printf "Missing metadata.yml for $folder, please provide metadata as described in the readme." 1>&2; exit_after=1);
    test $( ls -1 $folder/*.ipynb | wc -l ) != 1 -a ! -f $folder/executed_notebook.ipynb && (printf "Found more than one notebook in note $folder, only one notebook is allowed" 1>&2; exit_after=1);
    test $( ls -1 $folder/ | wc -l ) != 3 -a $( ls -1 $folder/ | wc -l ) != 4 -a $( ls -1 $folder/ | wc -l ) != 5 && (printf "Found more then 3 files in $folder, files are \n$( ls -1 $folder/ )" 1>&2; exit_after=1);
    test $exit_after == 1 && exit 3;
}

for folder in $( ls -d */ )
do
    printf "+++++++++++++++++++++++++++++++ \n";
    printf "Processing $folder...\n";
    printf "+++++++++++++++++++++++++++++++ \n";
    check_files $folder;
    #test $? != 0 && echo wtf;

    # install requirements
    pip install -r $reqs;
    printf "=============================== \n";
    
    if [ ! -f $folder/executed_notebook.ipynb ]; # Only run if not already:
    then
        echo Running notebook $notebook ...;
        python run_notebook.py $notebook;
        if [ "$TRAVIS_PULL_REQUEST" == "false"]; 
        then
            echo Adding executed notebook to github ...;
            git add $folder/executed_notebook.ipynb;
            git commit -m "new: ${SHA} Executed notebook $notebook";
        fi;
    else
        echo Notebook $notebook already run, not rerunning.;
    fi;
    if [ ! -f $folder/zenodo_upload.yml ]
    then
        python zenodo_upload_doi.py $ZENODO_ACCESS $ZENODO_ACCESS_TOKEN $metadata $notebook $reqs $folder;
        git add $folder/zenodo_upload.yml;
        git commit -m "new: $SHA Uploaded to zenodo $folder";
    fi;
    printf "\n";
done;


if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
    git push $SSH_REPO "$TRAVIS_BRANCH";
fi
