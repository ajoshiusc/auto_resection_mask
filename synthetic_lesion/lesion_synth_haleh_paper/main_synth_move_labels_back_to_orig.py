import torch

# from matplotlib.pyplot import plt
from nibabel.processing import conform
import matplotlib.pylab as plt

import nibabel as nib
import numpy as np
from monai.config import USE_COMPILED

from monai.networks.blocks import Warp, DVF2DDF

# Load the inpainted and original images

for subno in range(0, 21):

    middle_slice_idx = 64




    device = "cuda"


    if USE_COMPILED:
        warp_layer = Warp(3, padding_mode="zeros",mode="nearest").to(device)
    else:
        warp_layer = Warp("nearest", padding_mode="zeros").to(device)

    inpained_lab = nib.load(f'/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_inpainted_BrainSuite/Subject_{subno}_inpainted.svreg.label.nii.gz').get_fdata()

    aff = 2 * np.eye(4)
    aff[3, 3] = 1

    img = nib.Nifti1Image(np.int16(inpained_lab), aff)
    #img = conform(img, order=1)

    nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_moved_labels.nii.gz")


