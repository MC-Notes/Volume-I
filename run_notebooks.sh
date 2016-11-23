#!/bin/bash

if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
    REPO=`git config remote.origin.url`;
    SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:};
    SHA=`git rev-parse --verify HEAD`;
    ENCRYPTION_LABEL="$encrypted_a3a89bfc08a4";
    COMMIT_AUTHOR_EMAIL="ibinbei@gmail.com";
    openssl aes-256-cbc -K $encrypted_a3a89bfc08a4_key -iv $encrypted_a3a89bfc08a4_iv -in github_deploy.enc -out github_deploy -d;
    chmod 600 github_deploy;
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
    for notebook in $( ls $folder/*.ipynb )
    do
        if [ -a $folder/requirements.txt ]
        then
            pip install -r $folder/requirements.txt;
        fi;
        if [ ! -f $folder/executed_notebook.ipynb ] # Only run if not already:
        then
            echo Running notebook $notebook ...;
            python run_notebook.py $notebook;
            if [ "$TRAVIS_PULL_REQUEST" == "false" ]; 
            then
                echo Adding exectued notebook to github ...;
                git add $folder/executed_notebook.ipynb;
                git commit -m "new: ${SHA} Executed notebook $notebook";
            fi;
        else
            echo Notebook $notebook already run, not rerunning.;
        fi;
    done;
done;


if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
    git push $SSH_REPO "$TRAVIS_BRANCH";
fi
