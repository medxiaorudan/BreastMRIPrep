#!/bin/bash
# bash script to perform 2-stage coregistration
# optimized for coregistering same type of series within a subject but different
# sessions.

reference_dataset=$1
moving_dataset=$2
reg_be=$3
reg_le=$4
output_prefix=$5

reg_aladin -ref $reference_dataset \
        -flo $moving_dataset \
        -aff ${output_prefix}.aff.txt 
        -res moving_INT.nii

reg_f3d -ref $reference_dataset \
        -flo moving_INT.nii \
        -be $reg_be \
        -le $reg_le \
        -cpp ${output_prefix}.cpp.nii 
        -res ${output_prefix}.nii


# when applying transforms to mask (perhaps manual segmentation?)

reg_resample -ref ref.nii \
             -flo moving_seg.nii \
             -trans affine_transform.txt \
             -res moving_seg_INT.nii
reg_resample -ref ref.nii \
             -flo moving_seg_INT.nii \
             -trans deform_transform.nii \
             -res .nii


