#!/bin/bash

# perform 4D concatentation on a set of datasets 
# @args: 1. key: input search string, all files that contain "key" in their
#                filename will be concatenated together.
#        2. output: output filename

key=$1
output=$2

sessions=`find . -mindepth 2 -maxdepth 2 -type d`

orig_path=`pwd`
for s in $sessions; 
do
    echo "processing: $s"
    cd $s

	# Set IFS to the delimiter ("/")
	IFS='/' read -r -a filepath <<< "$s"
	subject_number=${filepath[1]}
	session_number=${filepath[2]}

    # grab all files to be concatenated
    files=`find . -type f -name "*$key*.nii" | sort | tr '\n' ' '` 
    if [ -z "$files" ]; then
        echo "no file found for this session, not doing anything ... "
    else
	    3dTcat -prefix ${subject_number}_${session_number}_${output}.nii $files
    fi

    cd $orig_path
done