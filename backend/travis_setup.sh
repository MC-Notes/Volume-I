#!/bin/sh
parse_yaml() {
    local prefix=$2
    local s
    local w
    local fs
    s='[[:space:]]*'
    w='[a-zA-Z0-9_]*'
    fs="$(echo @|tr @ '\034')"
    sed -ne "s|^\($s\)\($w\)$s:$s\"\(.*\)\"$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s[:-]$s\(.*\)$s\$|\1$fs\2$fs\3|p" "$1" |
    awk -F"$fs" '{
    indent = length($1)/2;
    vname[indent] = $2;
    for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=(\"%s\")\n", "'"$prefix"'",vn, $2, $3);
        }
    }' | sed 's/_=/+=/g'
}

function check_files {
   local exit_after=0

   # Do not run on gh-pages
   if [ $folder == "docs/" -o $folder == "backend/" ];
   then
     printf "Not running on gh-pages and backend\n" 1>&2; 
     return 0;
   fi;

   # Make sure required files exist (set in mc_config.yml)
   for f in ${required_files[@]};
   do
      test ! -f $folder/$f && (printf "Missing $f for $folder, please provide requirements as described in the readme (or empty file if no requirements).\n" 1>&2; exit_after=1);
   done;
   
   if [ $exit_after == 1 ];
   then
     return 2;
   else
     return 1;
   fi
}

eval $(parse_yaml mc_config.yml)
echo "Running backend environment $backend with workflow version $workflow_version"

eval $(parse_yaml backend/$backend/config.yml)
required_files=($metadata $reqs)
echo "Required Files: ${required_files[*]}"

echo "Downloading python for run"

if [ "$CI" == "true" ]; then
   wget https://github.com/mzwiessele/travis_scripts/raw/master/download_miniconda.sh;
   source download_miniconda.sh;
fi;

conda env create -f backend/conda_base.yml
source activate base