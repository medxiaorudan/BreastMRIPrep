# general imports
import numpy as np
import nibabel as nib

# packages to perform moco
import ants
import itk

import os
import shutil
import subprocess

def moco_ants(input_file, output_path, moco_type="deform"):
    '''
    perform motion correction using ANTs MRI package.

    Can choose between rigid, affine, and deform (SyN) algorithms.
    '''

    # check and clear tmp as *some* ants installation floods /tmp directory and
    # slows down the system.
    check_and_clear_tmp()
    
    dce = nib.load(input_file)
    dce_dat = dce.get_fdata()
    
    # grab first volume
    reference_image = ants.from_numpy(dce_dat[:,:,:,0])

    # using ANTs to perform motion correction
    for i in range(dce.shape[3]):
        a_vol = ants.from_numpy(dce_dat[:, :, :, i])
        
        if i == 0:
            output = dce_dat[:, :, :, 0]
            output = np.expand_dims(output, axis=3)
        else:
            if moco_type == "rigid":
                a_transform = ants.registration(reference_image, a_vol, "Rigid")
            elif moco_type == "affine":
                a_transform = ants.registration(reference_image, a_vol, "Affine")
            elif moco_type == "deform":
                a_transform = ants.registration(reference_image, a_vol, "SyN")
            else:
                print("unknown motion correction type entered, exiting.")
                return

            moco_a_vol = ants.apply_transforms(fixed=reference_image, 
                                               moving=a_vol, 
                                               transformlist=a_transform['fwdtransforms'])   
            moco_a_vol_np = moco_a_vol.numpy()
            moco_a_vol_np = np.expand_dims(moco_a_vol_np, axis=3)
            output = np.concatenate([output, moco_a_vol_np], axis=3)
    
    output_image_nib = nib.Nifti1Image(output, dce.affine, dce.header)
    nib.save(output_image_nib, output_path)
    return output_image_nib


def check_and_clear_tmp(threshold=90):
    """
    Check if the /tmp directory is almost full and clear it if it exceeds the threshold.
    
    :param threshold: The percentage (0-100) of used space in /tmp to trigger clearing.
                      Defaults to 90%.
    """
    # Get disk usage statistics for the /tmp directory
    total, used, free = shutil.disk_usage('/tmp')
    
    # Calculate the percentage of used space
    used_percentage = (used / total) * 100
    
    print(f"Used space in /tmp: {used_percentage:.2f}%")
    
    # If the used space exceeds the threshold, clear the /tmp directory
    if used_percentage > threshold:
        print(f"/tmp is almost full (>{threshold}%). Clearing the directory...")

        # Clear the /tmp directory
        try:
            # Using subprocess to clear the contents of /tmp directory
            subprocess.run(['rm', '-rf', '/tmp/*'], check=True)
            print("/tmp directory has been cleared.")
        except subprocess.CalledProcessError as e:
            print(f"Error clearing /tmp directory: {e}")
    else:
        print(f"/tmp is not full (below {threshold}%). No action taken.")


def moco_elastix(input_file, output_path):
    '''
    perform motion correction using SimpleElastix deformation registration.
    '''

    input_nib = nib.load(input_file)
    input_dat = input_nib.get_fdata()

    ref_vol = itk.GetImageFromArray(input_dat[:,:,:,0])
    ntimes = input_dat.shape[3]

    for i in range(ntimes):

        if i == 0:
            output = input_dat[:,:,:,0]
            output = np.expand_dims(output, axis=3) 
        else:
            moving_img = itk.GetImageFromArray(input_dat[:,:,:,i])

            registered_image, params = itk.elastix_registration_method(ref_vol, moving_img)
            mocoed_img = itk.GetArrayFromImage(registered_image)
            mocoed_img = mocoed_img.swapaxes(0, 2)

            tmp_output = np.expand_dims(mocoed_img, axis=3)
            output = np.concatenate([output, tmp_output], axis=3)


    output_image_nib = nib.Nifti1Image(output, input_nib.affine, input_nib.header)
    nib.save(output_image_nib, output_path)


def cvSIFT(img1, img2):
    '''
    Support function to calculate feature-based image similarity.
    Using OpenCV's SIFT algorithm to compare two images.
    ''' 
    # init SIFT
    sift=cv2.xfeatures2d.SIFT_create()
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    
    # find the keypoints and descriptors with SURF
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = bf.knnMatch(des1, des2, k=2)
    
    img_1 = cv2.drawKeypoints(img1, kp1, img1)
    img_2 = cv2.drawKeypoints(img2, kp2, img2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    return( len(good) / len(matches) )