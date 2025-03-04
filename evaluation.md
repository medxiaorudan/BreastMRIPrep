# Evaluating different parts of bmri-prep

## Motion Correction

Applicable to DCE (dynamic) sequence. If one opts to perform this step (should
always be the case unless the dynamic sequence is acquired using special
techniques such as keyhole/viewsharing where motion artefacts is not manifested
in the same manner.

## Methods

There are many methods to perform motion correction, mostly tailored towards
neuro MRI. In order to determine the optimal method to perform motion
correction, first we compared rigid, affine, and deformation registration using
the ANTs program package as the program package is known to offer great
performance in all three types of algorithms (REF), albeit mostly within neuro MRI. 

We then compared ANTs deformation registration algorithm with two other popular
motion correction programs packages that were capable of performing deformation
registration and evaluated their performances on breast MRI data: NiftyReg and
SimpleElastix.

## Evaluating motion correction results

Traditionally, visual inspection has been used to evaluate the results of motion
correction. Here we attempt to evaluate both through visual inspection and
utilize natural image processing metrics in order to determine the most suitable
method for motion correction.

In attempt to quantify motion correction results, structural similarity metric
($SSIM$) was used from the image-similarity-measures python package between two
slices. Feature-based similarity metrics was also attempted was on SURF (CITE),
but was ultimately abandoned (see Appendix). 

The first volume was used a reference and all subsequent volumes where compared
to the reference slice by and slice and then the metrics were averaged across
slices and time/dynamics. The motion correction metric is then taken as the
fractional improvement of motion correction based on the metrics:

$$M_{moco}=\frac{SSIM_{corrected}-SSIM_{reference}}{SSIM_{reference}}$$

## Results

($SSIM$) was used from the image-similarity-measures python package between two
The quantitative metrics doesn't provide much insight either. SSIM itself failed
to measure the high degree of similarity between volumes within the same dynamic
series. However, measurable differences can be quantified through our motion
correction metric. While generally showing less than 1% improvement overall
across all motion correction methods. Results are similar between affine and
rigid transformations while elastic transformations resulted in slightly better
metrics. This is expected since elastic transformation usually consist of affine
transformation prior to the elastic deformation fine-tuning step. Processing
time for rigid and affine motion correction is almost identical, while elastic
is considerably longer (figure). 



## Conclusion

Considering the generally small amount of motion within dynamic series, is it is
more reasonable to apply only rigid or affine transformations for motion
correction as visual inspection reveals no noticeable change between the
methods. Since any potential motion between axial slices of breast MRI cannot be
reasonably considered rigid motion, we deem affine transformations the most
suitable for motion correction breast MRI DCE dynamic series.

## 2. Coregistration

All other sequences should, after motion-correction where applicable, be
coregistered to the pre-contrast DCE reference volume.

## Coregistration methods

In essence, coregistration is very similar to motion-correction, with difference
being that once expects more noticeable differences in location between series
rather than within a single dynamic series. By this reasoning we may utilize
much of the evaluation results for motion-correction here. However, the
differences will impact how we approach coregistration. Deformation
coregistration algorithms are previously shown to be superior to rigid and
affine coregistration approaches. This applies more so in coregistration between
different series since more motion is expected. We evaluate all popular
deformation coregistration packages, SimpleElastix, ANTs, and NiftyReg, in
application on breast MR sequences to see which program package provides the
best results.

Breast MRI protocol, besides T1-weighted DCE dyanmic sequences, consists of
T2-weighted axial images, fat saturated and not, and secondarily DWI.
T2-weighted images might also be taken, though this is less common. The scope of
this evaluation is primarily focused on core sequences, so we will be evaluating
performance of coregistering axial T2-weighted images and DWI to the reference
volume.

Generally, we expect more motion from different series in a session than within
a single series, and more importantly, the SNR contrast, and even FOV is
different between different series, finetuning the registration parameter is
needed for robust results.

## Coregistration of longitudinal data

Two stage coregistration is available for coregistering (reference) series from
two different sessions for a patients with one other.

Movement, as well as FOV placement, is expected to vary more substantially
between sessions compared to series within a single session. However, since the
same type of series will be used as reference and moving series, contrast and
SNR will be much more similar with each other compared to the previous case.

Most deformation registration algorithms, including NiftyReg, performs first an
affine transformation and then applies the deformation algorithm on top by
default.  This is however an internal process and cannot be controlled
explicitly during call. At most the option is available to skip the affine
transformation. Testing showed that the affine tranformation is quite restricted
in performing bulk movements, which produces inadequate results when the FOV is
different, hence we propose a manual two-stage registration scheme where we
first perform affine transformation, then apply the deformation algorithm. The
transforms from the two-stage registration can be concatenated.

Optimzation of regularization parameters for the deformation algorithm indicate
that 6-8 times the default values results in good registration of breast tissue
structures. Note this is very much relaxed compared to coregistering different
series within the same session since feature detections is expected to be more
robust as the same type of series is used for reference, and series to be
registered.

## Results

Standard parameters produce unsatisfactory results (FIGURE).

When difficult series is involved such as DWI, where SNR and resolution is low,
and multiple sources of imaging artifacts are present to deform the anatomy even
further, perfect coregitration might not be possible.

Binary Search of regularization parameters showed that a 20-30 times the default
values prevents excessive smearing of breast tissue while deforming the anatomy
in accordance to the reference series.