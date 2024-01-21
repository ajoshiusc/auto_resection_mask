import torch

# from matplotlib.pyplot import plt
from nibabel.processing import conform
import matplotlib.pylab as plt

import nibabel as nib
import numpy as np

import SimpleITK as sitk
import numpy as np

def dilate_segmentation_mask(input_path, output_path, dilation_mm):
    # Read the segmentation mask
    original_mask = sitk.ReadImage(input_path)

    # Get the spacing (pixel dimensions) of the image
    spacing = np.array(original_mask.GetSpacing())

    # Calculate the number of pixels to dilate based on the given mm
    dilation_pixels = int(dilation_mm / spacing[0])

    # Create a structuring element for dilation
    dilation_size = [dilation_pixels] * original_mask.GetDimension()
    structuring_element = sitk.sitkBall
    dilator = sitk.BinaryDilateImageFilter()
    dilator.SetKernelType(structuring_element)
    dilator.SetKernelRadius(dilation_size)

    # Perform dilation
    dilated_mask = dilator.Execute(original_mask)

    # Write the dilated mask to NIfTI file
    sitk.WriteImage(dilated_mask, output_path)




# Load the inpainted and original images

for subno in range(27,32):

    if subno == 30:
        continue
    
    middle_slice_idx = 64

    m = torch.load(f"/deneb_disk/Inpainting_Lesions_Examples/synth_subjects/Subject_{subno}/lesion_mask.pt")
    aff = 2 * np.eye(4)
    aff[3, 3] = 1

    data = np.int16(m[0, 0].cpu().numpy()>0)
    data = np.flip(data, axis=1)
    data = np.maximum(data, 0.0)
    msk = nib.Nifti1Image(np.int16(data), aff)
    #img = conform(img, order=1)
    nib.save(msk, f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_lesion_mask.nii.gz")

    dilation_distance_mm = 10.0
    input_mask_path = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_lesion_mask.nii.gz"
    output_dilated_mask_path = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_lesion_dilated_{dilation_distance_mm}_mask.nii.gz"

    dilate_segmentation_mask(input_mask_path, output_dilated_mask_path, dilation_distance_mm)

    lesion_mask = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_lesion_mask.nii.gz"
    lesion_dilated_mask = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_lesion_dilated_{dilation_distance_mm}_mask.nii.gz"

    lesion = nib.load(lesion_mask).get_fdata()
    dlesion = nib.load(lesion_dilated_mask).get_fdata()

    lesion_neighborhood = np.logical_and(dlesion, np.logical_not(lesion))

    lesion_neighborhood_file = lesion_mask.replace('.nii.gz',f'_neighborhood_{dilation_distance_mm}.nii.gz')
    nib.save(nib.Nifti1Image(np.int16(lesion_neighborhood), nib.load(lesion_mask).affine), lesion_neighborhood_file)
