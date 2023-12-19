# This code creats a rondom lesion in ranom brain

import os
#from tarfile import tar_filter

#from param import output
from myutils import (
    smooth_3d_segmentation_mask,
    random_lesion_segmentation_carc,
    random_normal_subject_carc,
)

import nilearn.image as ni
import numpy as np

from nilearn.plotting import plot_roi, show, plot_anat

import random
from nibabel.processing import conform
import nibabel as nb
#42
random.seed(3)


brats_data_directory = "/scratch1/akrami/Data_train/Test/Brats21"
norm_data_directory = "/scratch1/akrami/Data_train"

out_dir = "/scratch1/ajoshi/auto_resection_out"

# Read a random lesion segmentation file from the BRATS dataset
random_normal_t1_tmp, sub_name = random_normal_subject_carc(
    norm_data_directory
)

random_normal_t1 = os.path.join(out_dir, sub_name + "_norm_rand_t1.nii.gz")

conform(nb.load(random_normal_t1_tmp),out_shape=(160, 192, 160)).to_filename(random_normal_t1)

random_normal_subject_mask = random_normal_t1


# Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1), 

# Read a random lesion segmentation file from the BRATS dataset
random_lesion_segmentation_tmp, random_lesion_t1_tmp = random_lesion_segmentation_carc(brats_data_directory)

random_lesion_segmentation = os.path.join(out_dir, sub_name + "_lesion_seg.nii.gz")
random_lesion_t1 = os.path.join(out_dir, sub_name + "_lesion_t1.nii.gz")

conform(nb.load(random_lesion_segmentation_tmp),order=0,out_shape=(160, 192, 160)).to_filename(random_lesion_segmentation)
conform(nb.load(random_lesion_t1_tmp),order=0,out_shape=(160, 192, 160)).to_filename(random_lesion_t1)


seg_data = np.float64(ni.load_img(random_lesion_segmentation).get_fdata())

pre_lesion_mask = np.uint16(seg_data == 1)
post_lesion_mask = np.uint16((seg_data == 1) | (seg_data == 4)) 
pre_lesion_nii = ni.new_img_like(random_normal_t1, pre_lesion_mask)
post_lesion_nii = ni.new_img_like(random_normal_t1, post_lesion_mask)

pre_lesion = os.path.join(out_dir, sub_name + "_pre_lesion.mask.nii.gz")
post_lesion = os.path.join(out_dir, sub_name + "_post_lesion.mask.nii.gz")

pre_lesion_nii.to_filename(pre_lesion)
post_lesion_nii.to_filename(
    os.path.join(out_dir, sub_name + "_post_lesion.mask.nii.gz")
)

# Smooth the segmentation mask
pre_lesion_smoothed = os.path.join(
    out_dir, sub_name + "_pre_lesion_smoothed.mask.nii.gz"
)
post_lesion_smoothed = os.path.join(
    out_dir, sub_name + "_post_lesion_smoothed.mask.nii.gz"
)
smooth_3d_segmentation_mask(pre_lesion, pre_lesion_smoothed, iterations=3)
smooth_3d_segmentation_mask(pre_lesion_smoothed, pre_lesion_smoothed, iterations=1)

smooth_3d_segmentation_mask(post_lesion, post_lesion_smoothed, iterations=3)
smooth_3d_segmentation_mask(post_lesion_smoothed, post_lesion_smoothed, iterations=1)

p = plot_roi(
    roi_img=pre_lesion_smoothed,
    bg_img=random_normal_t1,
    vmax=1,
    vmin=0,
    title="Random lesion segmentation on random normal subject t1",
)
p.savefig(os.path.join(out_dir, sub_name + "_pre_lesion_smoothed.mask.png"))
cut_coords = p.cut_coords.copy()
p.close()

p2 = plot_roi(
    roi_img=post_lesion_smoothed,
    bg_img=random_normal_t1,
    cut_coords=cut_coords,
    vmax=1,
    vmin=0,
    title="Random lesion segmentation on random normal subject t1",
)
p2.savefig(os.path.join(out_dir, sub_name + "_post_lesion_smoothed.mask.png"))
p2.close()


# create pre and post lesion masks for registration
pre = ni.load_img(pre_lesion_smoothed).get_fdata()
post = ni.load_img(post_lesion_smoothed).get_fdata()
t1_mask_data = ni.load_img(random_normal_subject_mask).get_fdata()
pre_lesion_mask = np.uint16((t1_mask_data > 0.1) & (pre < 0.5))
post_lesion_mask = np.uint16((t1_mask_data > 0.1) & (post < 0.5))

pre_lesion = os.path.join(out_dir, sub_name + "_pre_lesion.mask.nii.gz")
post_lesion = os.path.join(out_dir, sub_name + "_post_lesion.mask.nii.gz")

ni.new_img_like(random_normal_t1, pre_lesion_mask).to_filename(pre_lesion)
ni.new_img_like(random_normal_t1, post_lesion_mask).to_filename(post_lesion)


p = plot_roi(
    roi_img=pre_lesion,
    bg_img=random_normal_t1,
    vmax=1,
    vmin=0,
    cut_coords=cut_coords,
    title="Random lesion segmentation on random normal subject t1",
)
p.savefig(os.path.join(out_dir, sub_name + "_pre_lesion.mask.png"))
p.close()

p2 = plot_roi(
    roi_img=post_lesion,
    bg_img=random_normal_t1,
    vmax=1,
    vmin=0,
    cut_coords=cut_coords,
    title="Random lesion segmentation on random normal subject t1",
)
p2.savefig(os.path.join(out_dir, sub_name + "_post_lesion.mask.png"))
p2.close()


# Do the registration now

from warper_incompressible import Warper


pre2post_lesion = os.path.join(out_dir, sub_name + "_pre2post_lesion.mask.nii.gz")
ddf = os.path.join(out_dir, sub_name + "_pre2post_lesion_ddf.mask.nii.gz")
jac_file = os.path.join(out_dir, sub_name + "_pre2post_lesion_jac.mask.nii.gz")


if 1: #not os.path.isfile(pre2post_lesion):
    nonlin_reg = Warper()
    nonlin_reg.nonlinear_reg(
        target_file=post_lesion,
        moving_file=pre_lesion,
        output_file=pre2post_lesion,
        ddf_file=ddf,
        reg_penalty=1e-3,
        nn_input_size=64,
        lr=1e-3,
        max_epochs=3000,
        loss="mse",
        jacobian_determinant_file=jac_file,
    )


# apply the ddf to the t1 image

from warp_utils import apply_warp
from monai.transforms import LoadImage, EnsureChannelFirst

t1 = ni.load_img(random_normal_t1).get_fdata()
pre_lesion = ni.load_img(pre_lesion).get_fdata()
pre_lesion = np.uint16(pre > 0.5)
t1_avg = np.mean(t1[pre > 0.5])
t1[pre > 0.5] = t1_avg
t1 = ni.new_img_like(random_normal_t1, t1)

t1_with_pre_lesion = os.path.join(out_dir, sub_name + "_t1_with_pre_lesion.nii.gz")
t1.to_filename(t1_with_pre_lesion)

moving = LoadImage(image_only=True)(t1_with_pre_lesion)
moving = EnsureChannelFirst()(moving)[None]
target = LoadImage(image_only=True)(t1_with_pre_lesion)
target = EnsureChannelFirst()(target)[None]
ddf = LoadImage(image_only=True)(ddf)
ddf = EnsureChannelFirst()(ddf)[None]

moved = apply_warp(moving_image=moving, disp_field=ddf, target_image=target)

moved_ti_file = os.path.join(out_dir, sub_name + "_lesion_final_t1.nii.gz")

original_t1 = os.path.join(out_dir, sub_name + "_original_t1.nii.gz")

os.system(f"cp {random_normal_t1} {original_t1}")



import nibabel as nb

nb.save(
    nb.Nifti1Image(
        moved[0, 0].detach().cpu().numpy(), ni.load_img(random_normal_t1).affine
    ),
    moved_ti_file,
)


p = plot_roi(
    roi_img=post_lesion,
    bg_img=moved_ti_file,
    vmax=1,
    vmin=0,
    cut_coords=cut_coords,
    title="Random lesion segmentation on random normal subject t1",
)
p.savefig(os.path.join(out_dir, sub_name + "_pre2post_lesion_moved.mask.png"))
p.close()

plot_anat(random_normal_t1, cut_coords=cut_coords, title="Original CAMCAN T1",output_file=os.path.join(out_dir, sub_name + "_original_t1.png"),vmax=t1_avg*2,vmin=0)
plot_anat(t1_with_pre_lesion, cut_coords=cut_coords, title="CAMCAN T1 with lesion core from BRATS",output_file=os.path.join(out_dir, sub_name + "_t1_with_pre_lesion.png"),vmax=t1_avg*2,vmin=0)
plot_anat(moved_ti_file, cut_coords=cut_coords, title="CAMCAN T1 with expanded lesion core",output_file=os.path.join(out_dir, sub_name + "_moved_t1.png"),vmax=t1_avg*2,vmin=0)
plot_anat(random_lesion_t1, cut_coords=cut_coords, title="BRATS T1 with original lesion",output_file=os.path.join(out_dir, sub_name + "_original_lesion_t1.png"),vmax=t1_avg*4,vmin=0)

