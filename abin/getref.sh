#!/bin/bash

# Get the first volume in the DCE as reference.
# For all sessions:
# Take the first volume of <subject>_<session>_dce.nii, 
# and names it <subjet>_<session>_ref.nii

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
    file=`find . -type f -name "*dce.nii" | tr -d '\n'`
    file_count=$(echo "$file" | wc -l)

    if [ "$file_count" -eq 1 ]; 
    then
        echo "making" $file "the reference dataset"
	    3dcalc -a $file"[0]" -expr 'a' -prefix ${subject_number}_${session_number}_ref.nii
    else
        echo "not exactly one file found, do not know what to do so not doing anything."
    fi

    cd $orig_path
done