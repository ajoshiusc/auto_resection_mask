#!/usr/bin/env python3

from monai.utils import set_determinism
from monai.networks.nets import GlobalNet, LocalNet, RegUNet, unet
from monai.config import USE_COMPILED
from monai.networks.blocks import Warp, DVF2DDF
import torch

import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt
import torch
from monai.networks.blocks import Warp, DVF2DDF
#from monai.config import USE_COMPILED

device='cpu'

x = torch.load("/deneb_disk/Inpainting_Lesions_Examples/from_omar/In-painted_Lesion_Examples/example5.pt")

# save as nifti image

input_image = x[0].cpu().detach().numpy()

input_image = input_image[0,0,:,:,:]

aff = np.eye(4)
aff[0,0] = 2.0
aff[1,1] = 2.0
aff[2,2] = 2.0

nib.Nifti1Image(input_image, affine=None).to_filename('/deneb_disk/Inpainting_Lesions_Examples/from_omar/In-painted_Lesion_Examples/example5.nii.gz')



if USE_COMPILED:
    warp_layer = Warp(3, padding_mode="zeros", mode="nearest").to(device)
else:
    warp_layer = Warp("nearest", padding_mode="zeros").to(device)


image_moved = warp_layer(x[0][:,0:1,...], x[2])


nib.Nifti1Image(input_image, affine=None).to_filename('/deneb_disk/Inpainting_Lesions_Examples/from_omar/In-painted_Lesion_Examples/example5.nii.gz')



fig, axs = plt.subplots(3, 7, figsize=(21, 7))  # 3 rows for each image, 7 columns for the slices
middle_slice_idx = 80//2
slice_indices = [middle_slice_idx + i for i in range(-9, 10, 3)]  # Slices from middle_slice-9 to middle_slice+9 with step 3
for i, slice_idx in enumerate(slice_indices):
    # Plot for the first image (Inpainted) at each slice
    axs[0, i].imshow(x[0][0, 0, :, :, slice_idx].cpu(), cmap="gray",
                     vmin=x[0][0, 0, :, :, slice_idx].min(), vmax=x[0][0, 0, :, :, slice_idx].max())
    axs[0, i].set_title(f'Inpainted Slice {slice_idx}')
    axs[0, i].axis('off')  # Hide the axis
    # Plot for the second image (Original) at each slice
    axs[1, i].imshow(image_moved[0, 0, :, :, slice_idx].cpu(), cmap="gray",
                     vmin=image_moved[0, 0, :, :, slice_idx].min(), vmax=image_moved[0, 0, :, :, slice_idx].max())
    axs[1, i].set_title(f'Moved Slice {slice_idx}')
    axs[1, i].axis('off')  # Hide the axis
    # Plot for the second image (Original) at each slice
    axs[2, i].imshow(x[1][0, 0, :, :, slice_idx].cpu(), cmap="gray",
                     vmin=x[1][0, 0, :, :, slice_idx].min(), vmax=x[1][0, 0, :, :, slice_idx].max())
    axs[2, i].set_title(f'Original Slice {slice_idx}')
    axs[2, i].axis('off')  # Hide the axis
plt.tight_layout()
plt.show()