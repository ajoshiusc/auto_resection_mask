import torch
import os
# from matplotlib.pyplot import plt
from nibabel.processing import conform
import matplotlib.pylab as plt

import nibabel as nib
import numpy as np

# Load the inpainted and original images

for subno in range(27, 32):

    middle_slice_idx = 64

    x = torch.load(f"/deneb_disk/Inpainting_Lesions_Examples/synth_subjects/Subject_{subno}/lesion_deformation.pt")

    # Save the inpainted image as a nifti file
    aff = 2 * np.eye(4)
    aff[3, 3] = 1

    data = x[0].cpu().numpy()
    data = np.transpose(data,axes=(0,3,2,1))

    data = np.flip(data, axis=1+1)
    data[1] = -data[1]
    data = 0.5*data
    lesion_def = nib.Nifti1Image(np.transpose(data,axes=(1,2,3,0)), aff)
    #img = conform(img, order=1)
    nib.save(lesion_def, f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_lesion_def.map.nii.gz")

    from monai.config import USE_COMPILED

    from monai.networks.blocks import Warp, DVF2DDF


    device = "cuda"


    if USE_COMPILED:
        warp_layer = Warp(3, padding_mode="zeros",mode="bilinear").to(device)
    else:
        warp_layer = Warp("bilinear", padding_mode="zeros").to(device)


    if USE_COMPILED:
        warp_layer_nn = Warp(3, padding_mode="zeros",mode="nearest").to(device)
    else:
        warp_layer_nn = Warp("nearest", padding_mode="zeros").to(device)


    x = nib.load(f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_orig_wolesion.nii.gz")
    aff = x.affine
    x = x.get_fdata()

    map = torch.tensor(np.transpose(lesion_def.get_fdata(),axes=(3,0,1,2))[None]).float().to(device)
    x= torch.tensor(x[None,None]).float().to(device)
    wdata = warp_layer(x,map)[0,0].cpu().numpy()

    
    img = nib.Nifti1Image(np.int16(wdata), aff)
    #img = conform(img, order=1)

    nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_warp.nii.gz")

    lab_fname = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_orig_wolesion_BrainSuite/Subject_{subno}_orig_wolesion.svreg.label.nii.gz"

    if not os.path.exists(lab_fname):

        print(f"Label file {lab_fname} does not exist. Skipping.")
        continue

    lab = nib.load(lab_fname)
    lab = lab.get_fdata()
    lab = torch.tensor(lab[None,None]).float().to(device)
    wlab = warp_layer_nn(lab,map)[0,0].cpu().numpy()

    
    lab = nib.Nifti1Image(np.int16(wlab), aff)
    #img = conform(img, order=1)

    nib.save(lab, f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_warp.label.nii.gz")

    lesion_mask = nib.load(f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_lesion_mask.nii.gz")

    lab_with_lesion = wlab.copy()
    lab_with_lesion[lesion_mask.get_fdata() > 0] = 10000

    lab = nib.Nifti1Image(np.int16(lab_with_lesion), aff)

    nib.save(lab, f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_warp.withlesion.label.nii.gz")
   
