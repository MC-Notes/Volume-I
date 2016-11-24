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

for folder in $( ls -d */ )
do
    echo $folder
    echo +++++++++++++++++++++++++++++++;
    reqs=$folder/requirements.txt;
    metadata=$folder/metadata.yml;
    notebook=$( ls $folder/*.ipynb );
    if [ $( $notebook | wc -l ) != 1 ];
    then
        echo "Found more than one notebook in note $folder, only one notebook allowed";
        exit 3;
    elif [ $( ls -1 $folder | wc -l ) != 3 ];
    then
        echo "Found more than 3 files in note $folder, found files are: $( ls -1 $folder )";
        exit 4;
    fi;
    if [ -a reqs ];
    then
        pip install -r reqs;
        echo -----------------------------------;
    else
        echo "Missing requirements.txt for $notebook, please provide requirements as described in the readme (or empty file if no requirements).";
        exit 5;
    fi;
    if [ ! -f $metadata ];
    then
        echo "Missing metadata.yml for $notebook, please provide metadata as described in the readme.";
        exit 6;
    fi;
    if [ ! -f $folder/executed_notebook.ipynb ]; # Only run if not already:
    then
        echo Running notebook $notebook ...;
        python run_notebook.py $notebook;
        if [ "$TRAVIS_PULL_REQUEST" == "false" ]; 
        then
            echo Adding exectued notebook to github ...;
            git add $folder/executed_notebook.ipynb;
            git commit -m "new: ${SHA} Executed notebook $notebook";
            python zenodo_upload_doi.py $ZENODO_ACCESS $ZENODO_ACCESS_TOKEN $metadata $notebook $reqs
        fi;
    else
        echo Notebook $notebook already run, not rerunning.;
    fi;
    echo +++++++++++++++++++++++++++++++; 
done;


if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
    git push $SSH_REPO "$TRAVIS_BRANCH";
fi
