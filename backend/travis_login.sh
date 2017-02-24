if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
    if [ "$CI" == "true" ]; then
        REPO=`git config remote.origin.url`;
        SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:};
        SHA=`git rev-parse --verify HEAD`;
        ENCRYPTION_LABEL="$encrypted_a3a89bfc08a4";
        COMMIT_AUTHOR_EMAIL="ibinbei@gmail.com";
        openssl aes-256-cbc -K $encrypted_a3a89bfc08a4_key -iv $encrypted_a3a89bfc08a4_iv -in backend/secrets.tar.enc -out backend/secrets.tar -d
        tar xvf backend/secrets.tar;
        chmod 600 github_deploy;
        chmod 600 zenodo-access;
        eval `ssh-agent -s`;
        ssh-add github_deploy;
        ZENODO_ACCESS_TOKEN=`cat zenodo-access`;
        ZENODO_ACCESS=https://sandbox.zenodo.org/api/deposit/depositions;
        #git clone $REPO out
        #cd out
        git checkout $TRAVIS_BRANCH
        git config user.name "Travis CI";
        git config user.email "$COMMIT_AUTHOR_EMAIL";
    else
        # For local run of script:
        ZENODO_ACCESS_TOKEN=`cat zenodo-access`;
        ZENODO_ACCESS=https://sandbox.zenodo.org/api/deposit/depositions;
    fi;
fi;