import torch

# from matplotlib.pyplot import plt
from nibabel.processing import conform
import matplotlib.pylab as plt

import nibabel as nib
import numpy as np

# Load the inpainted and original images

sub='example8'
middle_slice_idx = 64
x = torch.load(f"/deneb_disk/Inpainting_Lesions_Examples/data/{sub}.pt")

# Save the inpainted image as a nifti file
aff = 2 * np.eye(4)
aff[3, 3] = 1

data = x[0][0, 0].cpu().numpy()
data = np.flip(data, axis=1)
data = 100.0*np.maximum(data, 0.0)
img = nib.Nifti1Image(data, aff)
img = conform(img, order=1)
nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/data/{sub}_inpainted.nii.gz")


data = x[1][0, 0].cpu().numpy()
data = np.flip(data, axis=1)
data = 100*np.maximum(data, 0.0)

img = nib.Nifti1Image(data, aff)
img = conform(img, order=1)

nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/data/{sub}_orig.nii.gz")
