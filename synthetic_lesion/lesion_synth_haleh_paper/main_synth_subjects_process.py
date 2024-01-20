import torch

# from matplotlib.pyplot import plt
from nibabel.processing import conform
import matplotlib.pylab as plt

import nibabel as nib
import numpy as np

# Load the inpainted and original images

for subno in range(27, 32):

    middle_slice_idx = 64

    x = torch.load(f"/deneb_disk/Inpainting_Lesions_Examples/synth_subjects/Subject_{subno}/All_results.pt")

    # Save the inpainted image as a nifti file
    aff = 2 * np.eye(4)
    aff[3, 3] = 1

    data = x[0][0, 0].cpu().numpy()
    data = np.flip(data, axis=1)
    data = 100.0*np.maximum(data, 0.0)
    img_inpainted = nib.Nifti1Image(np.int16(data), aff)
    #img = conform(img, order=1)
    nib.save(img_inpainted, f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_inpainted.nii.gz")

    from monai.config import USE_COMPILED

    from monai.networks.blocks import Warp, DVF2DDF


    device = "cuda"


    if USE_COMPILED:
        warp_layer = Warp(3, padding_mode="zeros",mode="nearest").to(device)
    else:
        warp_layer = Warp("nearest", padding_mode="zeros").to(device)


    if USE_COMPILED:
        warp_layer_lin = Warp(3, padding_mode="zeros",mode="bilinear").to(device)
    else:
        warp_layer_lin = Warp("bilinear", padding_mode="zeros").to(device)


    m = warp_layer_lin(x[0].float(), x[3].float())

    data = m[0, 0].cpu().numpy()
    data = np.flip(data, axis=1)
    data = 100*np.maximum(data, 0.0)

    img = nib.Nifti1Image(np.int16(data), aff)
    #img = conform(img, order=1)

    nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_moved.nii.gz")




    data = x[1][0, 0].cpu().numpy()
    data = np.flip(data, axis=1)
    data = 100*np.maximum(data, 0.0)

    img = nib.Nifti1Image(np.int16(data), aff)
    #img = conform(img, order=1)

    nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_orig.nii.gz")


    data = x[2][0, 0].cpu().numpy()
    data = np.flip(data, axis=1)
    data = 100*np.maximum(data, 0.0)

    img = nib.Nifti1Image(np.int16(data), aff)
    #img = conform(img, order=1)

    nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_before_inpainting.nii.gz")


    im_shape = x[1][0, 0].cpu().numpy().shape
    print(im_shape)

    data = x[3][0].cpu().numpy()
    data = np.flip(data, axis=2)
    data[1] *= -1

    img = nib.Nifti1Image(np.transpose(data,axes=(1,2,3,0)), aff)
    #img = conform(img, order=1)

    nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_ddf.nii.gz")


