import torch

# from matplotlib.pyplot import plt
from nibabel.processing import conform
import matplotlib.pylab as plt

import nibabel as nib
import numpy as np

# Load the inpainted and original images

for subno in range(11):

    middle_slice_idx = 64

    m = torch.load(f"/deneb_disk/Inpainting_Lesions_Examples/masks/Subject_{subno}_mask.pt")
    aff = 2 * np.eye(4)
    aff[3, 3] = 1

    data = m[0, 0].cpu().numpy()
    data = np.flip(data, axis=1)
    data = 100.0*np.maximum(data, 0.0)
    msk = nib.Nifti1Image(np.int16(data), aff)
    #img = conform(img, order=1)
    nib.save(msk, f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_lesion_mask.nii.gz")

