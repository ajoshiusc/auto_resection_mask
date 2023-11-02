"""
This module provides a function to process images.
"""

from warp_utils import apply_warp
import os
import tempfile

import nibabel as nib
from nilearn.plotting import plot_anat, plot_roi
from monai.transforms import EnsureChannelFirst, LoadImage
from aligner import Aligner, center_and_resample_images
import matplotlib.pyplot as plt
import numpy as np

# third party imports should be placed before standard library imports
# third party imports should be placed in alphabetical order
# standard library imports should be placed in alphabetical order


def get_possible_resect_mask(
    pre_mri_path,
    output_possible_resect_mask_path,
    bst_atlas_path,
    bst_atlas_labels_path,
):
    """
    Process images to create a resection mask.

    Args:
        pre_mri_path (str): Path to pre-operative MRI image.
        bst_atlas_path (str): Path to brain atlas.
        bst_atlas_labels_path (str): Path to brain atlas labels.
        output_possible_resect_mask_path (str): Path to output the resection mask.

    Returns:
        None
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary file paths
        temp_ddf = os.path.join(temp_dir, "temp_ddf.nii.gz")
        temp_out_img = os.path.join(temp_dir, "temp_out_img.nii.gz")
        temp_out_labels = os.path.join(temp_dir, "temp_out_labels.nii.gz")
        cent_bst_atlas = os.path.join(temp_dir, "cent_bst_atlas.nii.gz")
        cent_bst_atlas_labels = os.path.join(temp_dir, "cent_bst_atlas_labels.nii.gz")

        plot_anat(pre_mri_path, title="pre-mri")
        plot_roi(
            bst_atlas_labels_path,
            bg_img=bst_atlas_path,
            title="bst-atlas-labels on bst_atlas",
        )
        plot_roi(
            bst_atlas_labels_path,
            bg_img=pre_mri_path,
            title="bst-atlas-labels on pre-mri",
        )
        plt.show()

        center_and_resample_images(
            pre_mri_path,
            bst_atlas_path,
            centered_atlas=cent_bst_atlas,
            atlas_labels=bst_atlas_labels_path,
            centered_atlas_labels=cent_bst_atlas_labels,
        )

        plot_roi(
            cent_bst_atlas_labels,
            bg_img=pre_mri_path,
            alpha=0.25,
            title="centered bst-atlas",
        )

        affine_reg = Aligner()

        affine_reg.affine_reg(
            fixed_file=pre_mri_path,
            moving_file=cent_bst_atlas,
            output_file=temp_out_img,
            ddf_file=temp_ddf,
            loss="cc",
            nn_input_size=64,
            lr=1e-6,
            max_epochs=3000,
            device="cuda",
        )

        moving = LoadImage(image_only=True)(cent_bst_atlas_labels)
        moving = EnsureChannelFirst()(moving)

        target = LoadImage(image_only=True)(pre_mri_path)
        target = EnsureChannelFirst()(target)

        image_movedo = apply_warp(
            affine_reg.ddf[None,], moving[None,], target[None,], interp_mode="nearest"
        )

        nib.save(
            nib.Nifti1Image(
                image_movedo[0, 0].detach().cpu().numpy(), affine_reg.target.affine
            ),
            temp_out_labels,
        )

        plot_roi(temp_out_labels, bg_img=pre_mri_path, title="out-labels over pre-mri")
        plt.show()

        labels_to_zero = [3, 4, 5, 6]

        # Load the NIfTI file
        img = nib.load(temp_out_labels)

        # Get the data as a NumPy array
        data = img.get_fdata()

        # Set specific labels to zero
        for label in labels_to_zero:
            data[data == label] = 0

        data[data > 0] = 255

        # Create a new NIfTI image with modified data
        modified_img = nib.Nifti1Image(np.uint8(data), img.affine)

        # Save the modified NIfTI image to a new file
        nib.save(modified_img, output_possible_resect_mask_path)


if __name__ == "__main__":
    # Example usage
    pre_mri = "/deneb_disk/auto_resection/data_8_4_2023/sub-M1998N33/sMRI/sub-M1998N33-M1998N33_MRI.bse.nii.gz"
    bst_atlas = "/deneb_disk/auto_resection/bst_atlases/icbm_bst.nii.gz"
    bst_atlas_labels = "/deneb_disk/auto_resection/bst_atlases/icbm_bst.label.nii.gz"
    output_possible_resect_mask = "/deneb_disk/auto_resection/data_8_4_2023/sub-M1998N33/sMRI/sub-M1998N33-M1998N33_MRI.possible_resect.mask.nii.gz"

    get_possible_resect_mask(
        pre_mri, output_possible_resect_mask, bst_atlas, bst_atlas_labels
    )
