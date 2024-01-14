import torch

# from matplotlib.pyplot import plt
from nibabel.processing import conform
import matplotlib.pylab as plt

import nibabel as nib
import numpy as np

# Load the inpainted and original images

sub='example8'
middle_slice_idx = 64
x = torch.load(f"/deneb_disk/Inpainting_Lesions_Examples/from_omar/Final_Brats_Examples/{sub}.pt")

# Save the inpainted image as a nifti file
aff = 2 * np.eye(4)
aff[3, 3] = 1

data = x[0][0, 0].cpu().numpy()
data = np.flip(data, axis=1)
data = 100.0*np.maximum(data, 0.0)
img_inpainted = nib.Nifti1Image(np.int16(data), aff)
#img = conform(img, order=1)
nib.save(img_inpainted, f"/deneb_disk/Inpainting_Lesions_Examples/data_v3/{sub}_inpainted.nii.gz")

from monai.config import USE_COMPILED

from monai.networks.blocks import Warp, DVF2DDF


device = "cuda"


if USE_COMPILED:
    warp_layer = Warp(3, padding_mode="zeros",mode="nearest").to(device)
else:
    warp_layer = Warp("nearest", padding_mode="zeros").to(device)


m = warp_layer(x[0].float(), x[3].float())

data = m[0, 0].cpu().numpy()
data = np.flip(data, axis=1)
data = 100*np.maximum(data, 0.0)

img = nib.Nifti1Image(np.int16(data), aff)
#img = conform(img, order=1)

nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/data_v3/{sub}_moved.nii.gz")




data = x[1][0, 0].cpu().numpy()
data = np.flip(data, axis=1)
data = 100*np.maximum(data, 0.0)

img = nib.Nifti1Image(data, aff)
#img = conform(img, order=1)

nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/data_v3/{sub}_orig.nii.gz")

im_shape = x[1][0, 0].cpu().numpy().shape
print(im_shape)

data = x[3][0].cpu().numpy()
data = np.flip(data, axis=2)
data[1] *= -1

img = nib.Nifti1Image(np.transpose(data,axes=(1,2,3,0)), aff)
#img = conform(img, order=1)

nib.save(img, f"/deneb_disk/Inpainting_Lesions_Examples/data_v3/{sub}_ddf.nii.gz")




label = nib.load(f"/deneb_disk/Inpainting_Lesions_Examples/data/for_omar_12_10_2023/{sub}_inpainted/{sub}_inpainted.svreg.label.nii.gz")

img_inpainted

import nilearn.image as ni


label = ni.resample_to_img(label, img_inpainted, interpolation='nearest')
label.to_filename(f"/deneb_disk/Inpainting_Lesions_Examples/data_v3/{sub}_inpained_label.nii.gz")

label_data = np.float32(label.get_fdata())

# warp label data
label_data_warped = warp_layer(torch.tensor(label_data)[None,None].to(device), x[3])

label_warped = nib.Nifti1Image(label_data_warped[0,0].cpu().numpy(), aff)

label_warped.to_filename(f"/deneb_disk/Inpainting_Lesions_Examples/data_v3/{sub}_inpained_label_warped.nii.gz")







