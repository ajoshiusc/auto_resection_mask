# my_module.py

from aligner import Aligner, center_and_resample_images
from monai.transforms import LoadImage, EnsureChannelFirst
import nibabel as nib
from nilearn.plotting import plot_anat, plot_roi
import matplotlib.pyplot as plt
import matplotlib
from warp_utils import apply_warp
import tempfile
import os

def process_images(pre_mri, bst_atlas, bst_atlas_labels, cent_bst_atlas, cent_bst_atlas_labels):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary file paths
        temp_ddf = os.path.join(temp_dir, "temp_ddf.nii.gz")
        temp_out_img = os.path.join(temp_dir, "temp_out_img.nii.gz")
        temp_out_labels = os.path.join(temp_dir, "temp_out_labels.nii.gz")

        plot_anat(pre_mri, title='pre-mri')
        plot_roi(bst_atlas_labels, bg_img=bst_atlas, title='bst-atlas-labels on bst_atlas')
        plot_roi(bst_atlas_labels, bg_img=pre_mri, title='bst-atlas-labels on pre-mri')
        plt.show()

        center_and_resample_images(pre_mri, bst_atlas, centered_atlas=cent_bst_atlas,
                                   atlas_labels=bst_atlas_labels, centered_atlas_labels=cent_bst_atlas_labels)

        plot_roi(cent_bst_atlas_labels, bg_img=pre_mri, alpha=0.25, title='centered bst-atlas')

        affine_reg = Aligner()

        affine_reg.affine_reg(
            fixed_file=pre_mri,
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

        target = LoadImage(image_only=True)(pre_mri)
        target = EnsureChannelFirst()(target)

        image_movedo = apply_warp(affine_reg.ddf[None,], moving[None,], target[None,], interp_mode='nearest')

        nib.save(
            nib.Nifti1Image(image_movedo[0, 0].detach().cpu().numpy(), affine_reg.target.affine),
            temp_out_labels,
        )

        plot_roi(temp_out_labels, bg_img=pre_mri, title='out-labels over pre-mri')
        plt.show()


if __name__ == "__main__":
    # Example usage
    pre_mri = '/deneb_disk/auto_resection/data_8_4_2023/sub-M1998N33/sMRI/sub-M1998N33-M1998N33_MRI.bse.nii.gz'
    bst_atlas = '/deneb_disk/auto_resection/bst_atlases/icbm_bst.nii.gz'
    bst_atlas_labels = '/deneb_disk/auto_resection/bst_atlases/icbm_bst.label.nii.gz'
    cent_bst_atlas = 'cent_T1_brain_t1.nii.gz'
    cent_bst_atlas_labels = 'cent_T1_brain_t1.label.nii.gz'
    
    process_images(pre_mri, bst_atlas, bst_atlas_labels, cent_bst_atlas, cent_bst_atlas_labels)
