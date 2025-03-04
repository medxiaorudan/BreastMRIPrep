# imports 
import numpy as np
import nibabel as nib
from PIL import Image
import cv2
from torch import nn
import subprocess
from tqdm import tqdm
from image_similarity_measures.quality_metrics import *
from skimage.transform import resize as resize
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# general support functions for testing and evaluation
# ----------------------------------------------------
def find_files_matching_string(root_dir, match_string):
    '''
    find all files matching match_string in root_dir 

    designed for BRIDS dataset directories in mind
    '''
    matching_files = []
    # First level directories
    for first_level_dir in os.listdir(root_dir):
        first_level_path = os.path.join(root_dir, first_level_dir)
        if os.path.isdir(first_level_path):
            # Second level directories
            for second_level_dir in os.listdir(first_level_path):
                second_level_path = os.path.join(first_level_path, second_level_dir)
                if os.path.isdir(second_level_path):
                    # Check files in second level directory
                    for file_name in os.listdir(second_level_path):
                        if match_string in file_name:
                            matching_files.append(os.path.join(second_level_path, file_name))
    return matching_files

# functions for perform motion-correction and coregistration 
# ----------------------------------------------------------
'''
By directly, or indirectly, calling NiftyReg binaries.
Checks if NiftyReg binaries in currently in PATH.
'''
def moco_nrdef(moving, output, ref="", be='0.005', le='0.05', logfile=""):
    '''
    Perform motion correction using NiftyReg's deformation registration
    algorithm (SyN)
    
    Details: Calls bash-script niftyreg_moco.sh to do the actual work.
    
    Currently do not support custom regularization parameters (implement soon?).
    
    NiftyReg output is by default suppressed, or enter log filename in the
    logfile parameter to pipe both stdout and stderr outputs to file.
    '''
    print("Motion correction using NiftyReg's deformation registration algorithm ...") 

    # prep logs
    if logfile == "":
        stdo = subprocess.DEVNULL
        stde = subprocess.DEVNULL
    else:
        stde = subprocess.STDOUT
        stdo = open(logfile, 'w')

    # parsing binaries and input data 
    niftyreg_bin = subprocess.run(['which', 'reg_f3d'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    afni_calc = subprocess.run(['which', '3dinfo'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    afni_info = subprocess.run(['which', '3dinfo'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    if niftyreg_bin == "":
        print("NiftyReg binary not found, are you sure you have installed NiftyReg and added executable directory in your PATH variable?")
        return False
    if afni_info == "" or afni_calc == "":
        print("AFNI binary not found, are you sure you have installed AFNI and added executable directory in your PATH variable?")
        return False

    mov_nt = subprocess.run(['3dinfo', '-nt', moving], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    if mov_nt == 'NO_DSET':
        print('error! no dataset found! exiting ...')
        return
    mov_nt = int(mov_nt)

    if mov_nt <= 1:
        print('warning! series is a single volume and not a 3D+t series,')
        print('there is nothing to motion correct! exiting ...') 
        return

    if ref == "":
        print("No reference given, using first volume of moving to perform motion correction")
        subprocess.run(['3dcalc', '-a', str(str(moving)+'[0]'), '-expr', 'a', '-prefix', 'tref.nii'], check=True, stdout=stdo, stderr=stde)
        ref = 'tref.nii'
    elif ref == "average":
        print("using average moving series as reference for motion correction")
        subprocess.run(['3dTstat', '-mean', '-prefix', 'tref.nii', moving], check=True, stdout=stdo, stderr=stde)
        ref = 'tref.nii'
    else:  
        print("using specified dataset as reference for motion correction")
        if not os.path.exists(ref):
            print("error! cannot find reference file, exiting ...")
            return
        ref_nt = subprocess.run(['3dinfo', '-nt', ref], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
        ref_nt = int(ref_nt)
        if ref_nt > 1:
            print("warning! reference series is not a single volume, but a 4D series.")
            print("using the first volume as the reference volume.")
            subprocess.run(['3dcalc', '-a', str(str(ref)+'[0]'), '-expr', 'a', '-prefix', 'tref.nii'], check=True, stdout=stdo, stderr=stde)
            ref = 'tref.nii'

    for i in tqdm( range(mov_nt) ):
        subprocess.run(['3dcalc', '-a', str(str(moving)+'['+str(i)+']'), '-expr', 'a', '-prefix', 'tmp'+"{:04d}".format(i)+'.nii'], check=True, stdout=stdo, stderr=stde)
        command = [niftyreg_bin, 
                    '-ref', ref,
                    '-flo', 'tmp'+"{:04d}".format(i)+'.nii',
                    '-be', be,
                    '-le', le,
                    '-cpp', 'tmp_trf_'+str(i)+'.cpp.nii',
                    '-res', 'tmpout'+"{:04d}".format(i)+'.nii']
        subprocess.run(command, check=True, stdout=stdo, stderr=stde)

    tcat_cmd = "3dTcat tmpout*.nii -prefix " + str(output)
    subprocess.call(tcat_cmd, stdout=stdo, stderr=stde, shell=True)
    subprocess.call(['rm tmp*.nii'], shell=True)
    if os.path.exists('tref.nii'):
        subprocess.call(['rm tref.nii'], shell=True)
    print('... done!')
    return output

def coreg_nrdef(ref, moving, cpp_output, output, be='0.02', le='0.2', logfile=""):
    '''
    Perform intra-session, inter-series coregistration using NiftyReg's
    defomration coregistration algorithm (SyN)

    Direct call to NiftyReg using subprocess.

    Regularization parameters can be directly controlled from this function,
    otherwise default parameters tuned towards intra-session, inter-series will
    be used.
    
    NiftyReg output is by default suppressed, or enter log filename in the
    logfile parameter to pipe both stdout and stderr outputs to file.
    '''
    print("Intra-session, inter-series coregistration")
    print("Using NiftyReg SyN ...") 

    if logfile == "":
        stdo = subprocess.DEVNULL
        stde = subprocess.DEVNULL
    else:
        stde = subprocess.STDOUT
        stdo = open(logfile, 'w')
    
    niftyreg_bin = subprocess.run(['which', 'reg_f3d'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    afni_calc = subprocess.run(['which', '3dinfo'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    afni_info = subprocess.run(['which', '3dinfo'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    if niftyreg_bin == "":
        print("NiftyReg binary not found, are you sure you have installed NiftyReg and added executable directory in your PATH variable?")
        return False
    if afni_info == "" or afni_calc == "":
        print("AFNI binary not found, are you sure you have installed AFNI and added executable directory in your PATH variable?")
        return False

    ref_nt = subprocess.run(['3dinfo', '-nt', ref], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    mov_nt = subprocess.run(['3dinfo', '-nt', moving], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    ref_nt = int(ref_nt)
    mov_nt = int(mov_nt)
    if ref_nt > 1:
        print("warning! reference series is not a single volume, but a 4D series.")
        print("using the first volume as the reference volume.")
        subprocess.run(['3dcalc', '-a', str(str(ref)+'[0]'), '-expr', 'a', '-prefix', 'tref.nii'], check=True, stdout=stdo, stderr=stde)
        ref = 'tref.nii'
    if mov_nt > 1:
        print('warning! moving series is not a single volume, but a 4D series') 
        print('registering each volume independently to the reference volume.') 
        for i in tqdm( range(mov_nt) ):
            subprocess.run(['3dcalc', '-a', moving+'[0]', '-expr', 'a', '-prefix', 'tmp'+"{:04d}".format(i)+'.nii'], check=True, stdout=stdo, stderr=stde)
            command = [niftyreg_bin, 
                        '-ref', ref,
                        '-flo', 'tmp'+"{:04d}".format(i)+'.nii',
                        '-be', be,
                        '-le', le,
                        '-cpp', cpp_output,
                        '-res', 'tmpout'+"{:04d}".format(i)+'.nii']
            subprocess.run(command, check=True, stdout=stdo, stderr=stde)

        tcat_cmd = "3dTcat tmpout*.nii -prefix " + str(output)
        subprocess.call(tcat_cmd, shell=True)
        subprocess.call(['rm tmp*.nii'], shell=True)
        if ref_nt > 1:
            subprocess.call(['rm tref.nii'], shell=True)
        return
    command = [niftyreg_bin, 
        '-ref', ref,
        '-flo', moving,
        '-be', be,
        '-le', le,
        '-cpp', cpp_output,
        '-res', output]
    subprocess.run(command, check=True, stdout=stdo, stderr=stde)
    
    print("... done!") 
    return output

def coreg_nr2stage(ref, moving, transform_prefix, output, be='0.007', le='0.07', logfile=""):
    print("Inter-session, two-stage coregistration")

    if logfile == "":
        stdo = subprocess.DEVNULL
        stde = subprocess.DEVNULL
    else:
        stde = subprocess.STDOUT
        stdo = open(logfile, 'w')
    
    niftyreg_aladin = subprocess.run(['which', 'reg_aladin'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    niftyreg_f3d = subprocess.run(['which', 'reg_f3d'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    if niftyreg_f3d == "" or niftyreg_aladin == "":
        print("NiftyReg binary not found, are you sure you have installed NiftyReg and added executable directory in your PATH variable?")
        return False

    command1 = [niftyreg_aladin, 
                '-ref', ref,
                '-flo', moving,
                '-aff', transform_prefix+'.aff.txt',
                '-res', 'intermediate.nii']
    command2 = [niftyreg_f3d, 
                '-ref', ref,
                '-flo', 'intermediate.nii',
                '-be', be,
                '-le', le,
                '-cpp', transform_prefix+'.cpp.nii',
                '-res', output]

    subprocess.run(command1, check=True, stdout=stdo, stderr=stde)
    subprocess.run(command2, check=True, stdout=stdo, stderr=stde)
    subprocess.run(['rm', 'intermediate.nii'])
    print('...done!')
    return output

def apply_nr2stage(ref, moving, transform_prefix, output, logfile=""):
    print("Inter-session, apply two-stage coregistration")

    if logfile == "":
        stdo = subprocess.DEVNULL
        stde = subprocess.DEVNULL
    else:
        stde = subprocess.STDOUT
        stdo = open(logfile, 'w')

    niftyreg_resample = subprocess.run(['which', 'reg_resample'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    if niftyreg_resample == "":
        print("NiftyReg binary not found, are you sure you have installed NiftyReg and added executable directory in your PATH variable?")
        return False

    command1 = [niftyreg_resample, 
                '-ref', ref,
                '-flo', moving,
                '-trans', transform_prefix+'.aff.txt',
                '-res', 'intermediate.nii'] 
    command2 = [niftyreg_resample, 
                '-ref', ref,
                '-flo', 'intermediate.nii',
                '-trans', transform_prefix+'.cpp.nii',
                '-res', output] 

    print('running 1st stage (rough affine transformation) ...')
    subprocess.run(command1, check=True, stdout=stdo, stderr=stde)
    print('running 2nd stage (finetine deformation) ...')
    subprocess.run(command2, check=True, stdout=stdo, stderr=stde)
    subprocess.run(['rm', 'intermediate.nii'])
    print('...done!')
    return output

def npy_bbox_nii(input, reference, output, laterality='', logfile=""):
    # prep logs
    if logfile == "":
        stdo = subprocess.DEVNULL
        stde = subprocess.DEVNULL
    else:
        stde = subprocess.STDOUT
        stdo = open(logfile, 'w')

    if laterality == 'left':
        ref = load_nii(reference, laterality)
    elif laterality == 'right':
        ref = load_nii(reference, laterality)
        ref = np.flip(ref, 2)
    else:
        print('unknown laterality, exiting ...')
        return
    
    fullwidth = input.shape[2]
    midpoint = int(np.floor(fullwidth/2))
    if laterality == 'left':
        pred = input[:, :, 0:midpoint]
        msk = np.zeros(input.shape)
        msk[:, :, 0:midpoint] = pred
    elif laterality == 'right':
        pred = input[:, :, midpoint:fullwidth]
        msk = np.zeros(input.shape)
        msk[:, :, midpoint:fullwidth] = pred

    # perform bbox
    b = bbox3(pred)
    bbox = np.zeros(pred.shape)
    bbox[b[0]:b[1], b[2]:b[3], b[4]:b[5]] = ref[b[0]:b[1], b[2]:b[3], b[4]:b[5]]
    full_bbox = np.zeros(input.shape)
    if laterality == 'left':
        full_bbox[:, :, 0:midpoint] = bbox
    elif laterality == 'right':
        full_bbox[:, :, midpoint:fullwidth] = bbox

    npy_resize_nii(full_bbox, reference, "tmp.nii", isMask=False)    
    box_left_cmd = ["3dAutobox", "-noclust", "-prefix", output, "-npad", "10", "tmp.nii"]
    subprocess.run(box_left_cmd, check=True, stdout=stdo, stderr=stde)
    subprocess.call(["rm tmp.nii"], shell=True)

    return msk

def npy_resize_nii(input, reference, output, isMask=True):
    ref = nib.load(reference)
    ref_data = ref.get_fdata()
    swap_data = input.swapaxes(0, 2) 
    input_data = resize(swap_data, ref_data.shape)
    if isMask:
        input_data = np.where(input_data>0, 1, 0)
    nii_img = nib.nifti1.Nifti1Image(input_data, affine=ref.affine, header=ref.header)
    nib.save(nii_img, output)

def seg_vit(input, output_mask, output_mask_l, output_mask_r, output_box_l, output_box_r, logfile=""):
    '''
    Work function to perform breast masking using ViT

    Output many variants of the same thing

    output_mask = full bilateral breast mask at input resolution
    output_mask_l = left breast mask only at input resolution
    output_mask_r = right breast mask only at input resolution
    output_box_l = bounding-box cut of the left breast mask
    output_box_l = bounding-box cut of the right breast mask

    '''

    # prep logs
    if logfile == "":
        stdo = subprocess.DEVNULL
        stde = subprocess.DEVNULL
    else:
        stde = subprocess.STDOUT
        stdo = open(logfile, 'w')

    print("Performing deep-learning powered breast masking")
    print("Using Vision Transformer")
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained(r"doggywastaken/segformer-b0-finetuned-bmri-prep")
    pred = inference_on_nii_both(input, processor, model)
    '''
    ref_l = load_nii(input, 'left')
    ref_r = load_nii(input, 'right')
    ref_r = np.flip(ref_r, 2)
    
    fullwidth = pred.shape[2]
    midpoint = int(np.floor(fullwidth/2))
    pred_l = pred[:, :, 0:midpoint]
    
    npy_resize_nii(pred, input, output_mask)    
    npy_resize_nii(mask_l, input, output_mask_l)    
    npy_resize_nii(mask_r, input, output_mask_r)    

    # perform bbox
    bl = bbox3(pred_l)
    br = bbox3(pred_r)
    bbox_l = np.zeros(pred_l.shape)
    bbox_r = np.zeros(pred_r.shape)
    bbox_l[bl[0]:bl[1], bl[2]:bl[3], bl[4]:bl[5]] = ref_l[bl[0]:bl[1], bl[2]:bl[3], bl[4]:bl[5]]
    bbox_r[br[0]:br[1], br[2]:br[3], br[4]:br[5]] = ref_r[br[0]:br[1], br[2]:br[3], br[4]:br[5]]
    full_bbox_l = np.zeros(pred.shape)
    full_bbox_r = np.zeros(pred.shape)
    full_bbox_l[:, :, 0:midpoint] = bbox_l
    full_bbox_r[:, :, midpoint:fullwidth] = bbox_r

    npy_resize_nii(full_bbox_l, input, "tmp_l.nii", isMask=False)    
    npy_resize_nii(full_bbox_r, input, "tmp_r.nii", isMask=False)    

    box_left_cmd = ["3dAutobox", "-noclust", "-prefix", output_box_l, "-npad", "10", "tmp_l.nii"]
    box_right_cmd = ["3dAutobox", "-noclust", "-prefix", output_box_r, "-npad", "10", "tmp_r.nii"]
    subprocess.run(box_left_cmd, check=True, stdout=stdo, stderr=stde)
    subprocess.run(box_right_cmd, check=True, stdout=stdo, stderr=stde)
    subprocess.call(["rm tmp_*.nii"], shell=True)
    '''

    # saving stuff
    npy_resize_nii(pred, input, output_mask)

    msk_l = npy_bbox_nii(pred, input, output_box_l, 'left')
    msk_r = npy_bbox_nii(pred, input, output_box_r, 'right')

    npy_resize_nii(msk_l, input, output_mask_l)
    npy_resize_nii(msk_r, input, output_mask_r)

    print("... done!")
    return


# evaluation functions for moco and coreg
# ---------------------------------------
'''
Attempts to calculate evaluation metrics based on structural similarity
measures.
'''
def calc_ssim(input_dataset):
    '''
    support function for calculating ssim metric
    '''
    dce = nib.load(input_dataset)
    dce_dat = dce.get_fdata()

    nslices = dce_dat.shape[2]
    ntimes = dce_dat.shape[3]
    ssim_tab = np.empty([nslices, ntimes-1])
    for i in range(nslices):
        reference_image = prepImgForMetricCalc(dce_dat, i, 0)
        for j in range(ntimes-1):
            an_image = prepImgForMetricCalc(dce_dat, i, j+1)
            # calc ssim
            ssim_tab[i, j] = ssim(reference_image, an_image)
    return ssim_tab

def prepImgForMetricCalc(input_numpy_array, slice_index, volume_index=-1):
    '''
    Support function to convert numpy array from MRI images to more resemble a
    natural image (picture) in computation aspects in preparation for
    calculating image similarity metrics ssim (and fsim -- REMOVED).

    Performs range checking, and numeric continuity, as well as expanding
    dimensions.
    '''
    if volume_index == -1:
        tmp = input_numpy_array[:,:,slice_index]
    else:
        tmp = input_numpy_array[:,:,slice_index,volume_index]
    tmp = 255*tmp/(np.ptp(tmp)+1)
    tmp = np.ascontiguousarray(tmp, dtype=np.uint8)
    tmp = np.expand_dims(tmp, axis=-1)
    return tmp

def eval_moco_ssim(reference_dataset, moco_dataset):
    '''
    Calculates ssim metric for a motion corrected dataset against reference
    dataset.
    '''
    ref_moco_eval_params = calc_ssim(reference_dataset)
    mov_moco_eval_params = calc_ssim(moco_dataset)
    ref_ssim = np.mean(ref_moco_eval_params) 
    mov_ssim = np.mean(mov_moco_eval_params) 
    ssim_moco_metric = (mov_ssim-ref_ssim)/ref_ssim
    return ssim_moco_metric

def eval_coreg_ssim(ref, moving_before, moving_after):
    '''
    Calculates the ssim metric for coregistration against reference dataset.
    Currently only 3D volumes are supported.
    '''

    ref_nib = nib.load(ref)
    moving_before_nib = nib.load(moving_before)
    moving_after_nib = nib.load(moving_after)

    ref_data = ref_nib.get_fdata()
    moving_before_data = moving_before_nib.get_fdata()
    moving_after_data = moving_after_nib.get_fdata()

    print("DBG: reference data shape:", ref_data.shape)
    print("DBG: moving data (before coreg) shape:", moving_before_data.shape)
    print("DBG: moving data (after coreg) shape:", moving_after_data.shape)
    if ref_data.shape != moving_before_data.shape:
        print("Warning! reference and moving data shape is not the same!")
        print("Resampling")
        resampled_moving_before_data = resize(moving_before_data, ref_data.shape)     
        moving_before_data = resampled_moving_before_data

    nslices = ref_data.shape[2]
    ssim_slices = np.zeros(nslices)
    for i in range(nslices):
        reference_image = prepImgForMetricCalc(ref_data, i, -1)
        moving_before_image = prepImgForMetricCalc(moving_before_data, i, -1)
        moving_after_image = prepImgForMetricCalc(moving_after_data, i, -1)

        ssim_ref = ssim(reference_image, moving_before_image)
        ssim_reg = ssim(reference_image, moving_after_image)

        ssim_slices[i] = (ssim_reg-ssim_ref)/ssim_ref

    return np.mean(ssim_slices)


# general utility functions for segmentation
# ------------------------------------------

def load_nii(filename, left_or_right="left", out_size=(112, 512, 512), is_mask=False):
    nii_file = nib.load(filename)
    nii_data = nii_file.get_fdata() 
    nii_data = nii_data.swapaxes(0, 2)
    if is_mask:
        nii_data = np.where(nii_data>0, 0, 1)
    
    if nii_data.shape != out_size:
        nii_data = resize(nii_data, out_size)

    nc = nii_data.shape[2]
    mp = int(np.floor(nc/2))
    if left_or_right == "left":
        return( nii_data[:, :, 0:mp] )
    elif left_or_right == "right":
        return( np.flip(nii_data[:, :, mp:nc], 2) )
    else:
        print("Error! Invalid left/right laterality!")
        return( None )

def unload_nii(master_filename, data, output_filename):
    data = data.swapaxes(0, 2)

    master_file = nib.load(master_filename)
    master_data = master_file.get_fdata()
    master_shape = master_data.shape

    if data.shape != master_shape:
        resize(data, master_shape)

    output_image_nib = nib.Nifti1Image(data, np.eye(4))
    nib.save(output_image_nib, output_filename)

def save_image_stack(input_data, ismask, output_prefix, extension, truncate=10):
    n_images = input_data.shape[0] 
    for i in range(truncate, n_images-truncate):
        a_dat = input_data[i, :, :]
        output_name = str(output_prefix)+'_slc'+str(i).zfill(3)+extension
        if not ismask:
            cv2.imwrite(output_name, cv2.merge((a_dat, a_dat, a_dat)))
        else:
            cv2.imwrite(output_name, a_dat)

def save_nii_stack(filename, laterality, ismask=False, tr=10):
    tt = load_nii(filename, laterality, is_mask=ismask)
    if tt is None:
        print("got empty. skipping save ...")
        return 
    else:
        data_tag = filename.split('/')[1]
        if ismask is False:
            pref = "data_" + laterality + "_" + data_tag
            save_image_stack(tt, ismask, pref, '.jpg', truncate=tr)
        else:
            pref = "mask_" + laterality + "_" + data_tag
            save_image_stack(tt, ismask, pref, '.jpg', truncate=tr)

# bounding box calculators
# ------------------------

def bbox3(img):
    a = np.where(img>0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1]), np.min(a[2]), np.max(a[2])
    return(bbox)

def bbox3_mask(img):
    bbox = bbox3(img)
    mask = np.zeros_like(img)
    mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = 1
    return mask

def bbox3_masked(img, data, expand_box_px=0):
    bbox = bbox3(img)
    mask = data[(bbox[0]-expand_box_px):(bbox[1]+expand_box_px), 
                (bbox[2]-expand_box_px):(bbox[3]+expand_box_px), 
                (bbox[4]-expand_box_px):(bbox[5]+expand_box_px)]
    return mask

        

# support functions for jpg-based inference for ViT
# -------------------------------------------------

def inference_on_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1], # (height, width)
        mode='bilinear',
        align_corners=False
    )

    # Second, apply argmax on the class dimension
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    return( pred_seg )

def gscale_to_3channel(input):
    exi = np.expand_dims(input, axis=2)
    return( np.concatenate([exi, exi, exi], axis=2) )

def inference_on_nii(a_nii, processor, model, laterality="left"):
    nii_stack = load_nii(a_nii, laterality)
    n_slices = nii_stack.shape[0]

    for i in tqdm( range(0, n_slices) ):
        slice_1ch= nii_stack[i,:,:]
        slice_3ch = gscale_to_3channel(slice_1ch)
        im = Image.fromarray(slice_3ch.astype(np.uint8))

        apred = inference_on_image(im, processor, model).detach().numpy()
        apred = np.expand_dims(apred, axis=0)

        if i == 0:
            pred_full = apred
        else:
            pred_full = np.concatenate([pred_full, apred], axis=0)

    return(pred_full)

def inference_on_nii_both(a_nii, processor, model):
    pred_l = inference_on_nii(a_nii, processor, model, 'left')
    pred_r = inference_on_nii(a_nii, processor, model, 'right')
    return( np.concatenate([pred_l, np.flip(pred_r, 2)], axis=2) )

def inference_on_image_dir(directory, stack_grep):
    files = []
    import glob, os
    for f in glob.glob( os.path.join(directory, stack_grep), recursive=True):
        files.append(f)
    files = sorted(files)

    for i in tqdm( range(len(files)) ):
        pred_img_path = files[i]
        an_img = Image.open(pred_img_path)
        pred = inference_on_image(an_img).detach().numpy()
        pred = np.expand_dims(pred, axis=0)
        if i == 0:
            pred_full = pred
        else:
            pred_full = np.concatenate([pred_full, pred], axis=0)

    ppf = np.pad(pred_full, ((20,20),(0, 0),(0, 0)) )
    ppf_inv = np.where((ppf==0)|(ppf==1), ppf^1, ppf)
    return ppf_inv