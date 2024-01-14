import torch

# from matplotlib.pyplot import plt
from nibabel.processing import conform
import matplotlib.pylab as plt

import nibabel as nib
import numpy as np

# Load the inpainted and original images

sub='example13'
middle_slice_idx = 64
x = torch.load(f"/deneb_disk/Inpainting_Lesions_Examples/from_omar/In-painted_Lesion_Examples/{sub}.pt")

# Save the inpainted image as a nifti file
aff = 2 * np.eye(4)
aff[3, 3] = 1

data = x[0][0, 0].cpu().numpy()
data = np.flip(data, axis=1)
data = 100.0*np.maximum(data, 0.0)
img = nib.Nifti1Image(data, aff)
img = conform(img, order=1)
nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/data_v2/{sub}_inpainted.nii.gz")


from monai.config import USE_COMPILED

from monai.networks.blocks import Warp, DVF2DDF


device = "cpu"


if USE_COMPILED:
    warp_layer = Warp(3, padding_mode="zeros").to(device)
else:
    warp_layer = Warp("bilinear", padding_mode="zeros").to(device)


m = warp_layer(x[0][:,0:1,...], x[2])

data = m[0, 0].cpu().numpy()
data = np.flip(data, axis=1)
data = 100*np.maximum(data, 0.0)

img = nib.Nifti1Image(data, aff)
img = conform(img, order=1)

nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/data_v2/{sub}_moved.nii.gz")




data = x[1][0, 0].cpu().numpy()
data = np.flip(data, axis=1)
data = 100*np.maximum(data, 0.0)

img = nib.Nifti1Image(data, aff)
img = conform(img, order=1)

nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/data_v2/{sub}_orig.nii.gz")
