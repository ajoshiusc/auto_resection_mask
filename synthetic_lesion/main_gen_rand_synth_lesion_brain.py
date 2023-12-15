# This code creats a rondom lesion in ranom brain

import os
from myutils import (
    smooth_3d_segmentation_mask,
    random_lesion_segmentation,
    random_normal_subject,
)

import nilearn.image as ni
import numpy as np

from nilearn.plotting import plot_roi, show

import random

random.seed(42)


brats_data_directory = "/ImagePTE1/ajoshi/data/BRATS2018/Training/HGG"
norm_data_directory = "/ImagePTE1/ajoshi/data/camcan_preproc"

out_dir = "/deneb_disk/auto_resection/lesion_masks"

# Read a random lesion segmentation file from the BRATS dataset
random_normal_t1, random_normal_subject_mask, sub_name = random_normal_subject(
    norm_data_directory
)


# Read a random lesion segmentation file from the BRATS dataset
random_lesion_segmentation = random_lesion_segmentation(brats_data_directory)

seg_data = np.uint16(ni.load_img(random_lesion_segmentation).get_fdata())

pre_lesion_mask = np.uint16(seg_data == 1)
post_lesion_mask = np.uint16(seg_data > 0)
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
    vmax=1,vmin=0,
    title="Random lesion segmentation on random normal subject t1",
)
p.savefig(os.path.join(out_dir, sub_name + "_pre_lesion_smoothed.mask.png"))
cut_coords = p.cut_coords.copy()
p.close()

p2 = plot_roi(
    roi_img=post_lesion_smoothed,
    bg_img=random_normal_t1,
    cut_coords=cut_coords,
    vmax=1,vmin=0,
    title="Random lesion segmentation on random normal subject t1",
)
p2.savefig(os.path.join(out_dir, sub_name + "_post_lesion_smoothed.mask.png"))
p2.close()


# create pre and post lesion masks for registration
pre = ni.load_img(pre_lesion_smoothed).get_fdata()
post = ni.load_img(post_lesion_smoothed).get_fdata()
t1_mask_data = ni.load_img(random_normal_subject_mask).get_fdata()
pre_lesion_mask = np.uint16((t1_mask_data>0) & (pre<0.5))
post_lesion_mask = np.uint16((t1_mask_data>0) & (post<0.5))

pre_lesion = os.path.join(out_dir, sub_name + "_pre_lesion.mask.nii.gz")
post_lesion = os.path.join(out_dir, sub_name + "_post_lesion.mask.nii.gz")

ni.new_img_like(random_normal_t1, pre_lesion_mask).to_filename(pre_lesion)
ni.new_img_like(random_normal_t1, post_lesion_mask).to_filename(post_lesion)


p = plot_roi(
    roi_img=pre_lesion,
    bg_img=random_normal_t1,
    vmax=1,vmin=0,
    cut_coords=cut_coords,
    title="Random lesion segmentation on random normal subject t1",
)
p.savefig(os.path.join(out_dir, sub_name + "_pre_lesion.mask.png"))
p.close()

p2 = plot_roi(
    roi_img=post_lesion,
    bg_img=random_normal_t1,
    vmax=1,vmin=0,
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
    jacobian_determinant_file=jac_file)
