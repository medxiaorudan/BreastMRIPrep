# Unified preprocessing pipeline for breast MRI data.

bMRIPrep is a breast magnetic resonance imaging (brastMR) data preprocessing
pipeline. It provides minimal preprocessing steps which includes motion
correction (where applicable), coregistration, and breast segmentation.

bMRIPrep is designed to provide outputs that can be easily ingested to a
variety to deep-learning networks with no further preprocessing (not including
specific formats for data ingestation).

bMRIPrep uses a combination of tools from well-known software packages, mostly
originating from functional MRI analysis pipelines due to their historic
prominence and continued active research and development by the authors of
these software packages and tools. Though due to the dissimilar nature between
breastMR and fMRI, only smaller parts of from these software packages were used
and adapted in fitting to the current aplication.


# Setup

## Requirements

Install required python packages

        pip install -r <path to bmri-ai-prep>/00_support_functions/requirements.txt

Install AFNI using the corresponding easy to follow guide
(https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/install_instructs/index.html). 

Note AFNI python wrappers is not used here, but pure AFNI binaries are required.

Make sure AFNI binaries is in your *PATH* variable. To test this, call a AFNI program directly after opening up a new terminal of your choice, for example:

        3dinfo -h

If you see a massively detailed help print out, then you should be good to go!

## Check environment

Run *runme_...ipynb* example scripts in sections 02-04 to ensure different
preprocessing steps can be run correctly, as well as familiarize oneself with
the command calls. *bmri-ai-prep* should all run correctly if all examples
scripts can be run without errors.

# Input data format

The input data format for bMRIPrep closely resembles The Brain Imaing Data
Structure ([BIDS][1], https://bids.neuroimaging.io/), the a few key differences.
Unlike brain imaging data, breast MRI data is not as varied in sequence types
and standards such as the [BIRADS][2] MRI protocol recommendations do exist,
hence subcategories for functional and anatoimcal data (*func/* and *anat/*
directories within each session) is skipped here, instead replaced by the
individual sequence names typical of a breast MRI examination.

* Default orientation is Axial unless otherwise specified.

* All series with more than 1 volume assumed to be concatenated to 4D *.nii files.

* Reference dataset should be a separate 3D *.nii file assumed to be a copy of a
volume in a sequence in the protocol. For example if the first dynamic of the
DCE series is taken as reference, then ...ref.nii should be a copy of the first
volume of ...dce.nii 

Example file structure and naming can be found in the *01_formatbids* directory.

## Details

Subject number assumed to be 3 digits, and session number 2, with zero padding
prior to small number for full digit representation. 

Using subject 1, session 2 as demonstrative example. Base sequence types are
always small letters while subtype description may contain capital letters for
easier distinction between words in label. All labels should optimally be
3-4 letter abbrieviations.

Currently supported types are:

* sub-001_sub-02_dce.nii for dynamic contrast enhanced series.

* sub-001_sub-02_dwi.nii for diffusion-weighted series.

* sub-001_sub-02_t1.nii for T1-weighted series

* sub-001_sub-02_t2.nii for T2-weighted series

Modifiers such as specific sequence types such as Dixon is labelled as:

* sub-001_sub-02_t2_<DixInP/DixOuP/DixWat/DixFat>.nii for Dixon in-phase,
  out-phase, water, and fat respectively.

Slice orientation other than Axial is labelled directly behind base sequence
type, for example:

* sub-001-sub-02_t1Sag.nii for sagittal T1-weighted series

* sub-001-sub-02_t2SagR.nii for right breast only sagittal T2-weighted series

# Main preprocessing pipeline

The main preprocessing pipeline consists mainly of the *PreProc* class, which
takes a input path of a series (*.nii*) file and performs path parsing. The
parses relevant information such as subject, session, and series type from the
naming scheme and determines relevant directory paths associated with the series
in order to establish which preprocessing steps to take and then constructs a
preprocessing pipeline in the form a *baip.txt* file (which is actually a *json* file). 

## baip.txt files

The class then will parse the *baip.txt* file and perform the prescribed
preprocessing steps using functions calls availble as examples in the various
sub-directories (sections 2-4). The *baip.txt* files are created but not
overwritten if one exists already, and each completed step for a series will be
updated in its *baip.txt* file, which allows for easy troubleshooting and allows
for long runs *on large datasets to continue when unexpectedly halted for
whatever reason. One may also construct all *baip.txt* files quickly, inspect
(or edit), them manually, and run them separately to have manual controll over
which preprocessing steps are performed for individual series. 


# References

[1]: <https://www.nature.com/articles/sdata201644> "The brain imaging data
structure, a format for organizaing and describing outputs of neuroimaging
experiments"

[2]: <https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Bi-Rads>

