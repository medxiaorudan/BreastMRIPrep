#!/bin/bash

# Converts dcm to nifti files.

subject_dcms=`find . -mindepth 2 -maxdepth 2 -type d`
home_path=`pwd`

for s in $subject_dcms; 
do
	echo "processing $s"
	cd $s

	dcm2niix_afni .

	cd $home_path

done
