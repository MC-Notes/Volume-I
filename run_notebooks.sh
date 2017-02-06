#!/bin/bash

if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
    if [ "$CI" == "true" ]; then
        REPO=`git config remote.origin.url`;
        SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:};
        SHA=`git rev-parse --verify HEAD`;
        ENCRYPTION_LABEL="$encrypted_a3a89bfc08a4";
        COMMIT_AUTHOR_EMAIL="ibinbei@gmail.com";
        openssl aes-256-cbc -K $encrypted_a3a89bfc08a4_key -iv $encrypted_a3a89bfc08a4_iv -in secrets.tar.enc -out secrets.tar -d
        tar xvf secrets.tar;
        chmod 600 github_deploy;
        chmod 600 zenodo_access;
        eval `ssh-agent -s`;
        ssh-add github_deploy;
        echo $REPO
        #git clone --branch=$TRAVIS_BRANCH $REPO out
        #cd out
        git config user.name "Travis CI";
        git config user.email "$COMMIT_AUTHOR_EMAIL";
    fi;
    ZENODO_ACCESS_TOKEN=`cat zenodo-access`;
    ZENODO_ACCESS=https://sandbox.zenodo.org/api/deposit/depositions;
fi;

function check_files {
    folder=$1
    reqs=$folder/requirements.txt;
    metadata=$folder/metadata.yml;
    notebook="$folder/$( ls $folder | grep -v 'executed_notebook.ipynb$' | grep '.ipynb$' )"
    local exit_after=0
    
    # Do not run on gh-pages
    if [ $folder == "docs/" ];
    then
        printf "Not running on gh-pages\n" 1>&2; 
        return 0;
    fi;
    
    # Make sure files exist
    test ! -f $reqs && (printf "Missing requirements.txt for $folder, please provide requirements as described in the readme (or empty file if no requirements).\n" 1>&2; exit_after=1);
    test ! -f $metadata && (printf "Missing metadata.yml for $folder, please provide metadata as described in the readme.\n" 1>&2; exit_after=1);
    #test $( ls -1 $folder/*.ipynb | wc -l ) != 1 -a ! -f $folder/executed_notebook.ipynb && (printf "Found more than one notebook in note $folder, only one notebook is allowed\n" 1>&2; exit_after=1);
    #test $( ls -1 $folder/ | wc -l ) != 3 -a $( ls -1 $folder/ | wc -l ) != 4 -a $( ls -1 $folder/ | wc -l ) != 5 && (printf "Found more then 3 files in $folder, files are \n$( ls -1 $folder/ )\n" 1>&2; exit_after=1);
    if [ $exit_after == 1 ];
    then
        return 0;
    else
        return 1;
    fi
}

for folder in $( ls -d */ )
do
    printf "+++++++++++++++++++++++++++++++ \n";
    printf "Processing $folder...\n";
    check_files $folder;
    test $? == 0 && continue;
    
    if [ ! -f $folder/executed_notebook.ipynb ] || [ ! -f $folder/executed_notebook.md ]; # Only run if not already:
    then
        # install requirements
        pip install -r $reqs;
        #printf "\n";
        echo Running notebook $notebook ...;
        python run_notebook.py $notebook;
        if [ "$TRAVIS_PULL_REQUEST" == "false" ]; 
        then
            echo "Adding executed notebook to github ...";
            git add $folder/executed_notebook.ipynb;
            git add $folder/executed_notebook.md;
            git commit -m "new: ${SHA} Executed notebook $notebook";
        fi;
    else
        #printf "\n";
        echo $notebook already run, not rerunning.;
    fi;
    if [ ! -f $folder/zenodo_upload.yml ];
    then
        if [ "$TRAVIS_PULL_REQUEST" == "false" ]; 
        then
            #printf "\n";
            echo Uploading $folder to zenodo;
            #python zenodo_upload_doi.py $ZENODO_ACCESS $ZENODO_ACCESS_TOKEN $metadata $folder/executed_notebook.ipynb $reqs $folder;
            #git add $folder/zenodo_upload.yml;
            #git commit -m "new: $SHA Uploaded to zenodo $folder";
        fi;
    else
        #printf "\n";
        printf "$folder already uploaded\n"; # as \n\n$( cat $folder/zenodo_upload.yml )\n";
    fi;
    printf "+++++++++++++++++++++++++++++++ \n\n";
done;

if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
    if [ "$CI" == "true" ]; then
        if [ -z `git diff --exit-code` ]; then
            echo "No changes to the output on this push; exiting."
            exit 0
        fi
        git push $SSH_REPO "$TRAVIS_BRANCH";
    else
        echo Updated tree, see git status for details.;
    fi;
fi
