#!/bin/bash

wget -O mc_download_backend.sh https://github.com/MC-Notes/backend/raw/master/mc_download_backend.sh
source mc_download_backend.sh
source backend/travis_setup.sh
source backend/travis_login.sh