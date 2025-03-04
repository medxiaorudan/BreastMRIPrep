#!/bin/bash

# renames files that matches "key" and renames them according to "affix"
# newly renamed files will be renamed <subject>_<session>_<affix>.<nii/json>

# WARNING! 
# that this script does not count the number of matches, so if multiple files
# matches "key" in a session, this script will seriously mess up your data!

# @args: 1. key: string match (contains, case sensitive).
#		 2. affix: string affix of the new filename.

key=$1
affix=$2

files=`find . -mindepth 3 -maxdepth 3 -type f -name "*${1}*.nii"`

for f in $files; 
do

	# split input into basename and fiepath
	bn=$(basename "$f")
	fp=$(dirname "$f")

	# split basename into name and extension
	name="${bn%.*}"
	extension="${bn##*.}"

	# Set IFS to the delimiter ("/")
	IFS='/' read -r -a filepath <<< "$f"
	subject_number=${filepath[1]}
	session_number=${filepath[2]}

	mv $f ${fp}/${subject_number}_${session_number}_${affix}.nii
	mv $fp/$name.json ${fp}/${subject_number}_${session_number}_${affix}.json

done
