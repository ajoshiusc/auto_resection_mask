import os
import numpy as np
import nibabel as nb
from nilearn.plotting import plot_anat, find_xyz_cut_coords
from matplotlib import pyplot as plt


def generate_postop_overlay(preop_mri, postop_mri):
    subid = os.path.basename(postop_mri).split("_")[0]

    if os.path.isfile(preop_mri) and os.path.isfile(postop_mri):
        if os.path.isfile(postop_mri.replace(".nii.gz", ".resection.mask.nii.gz")):
            resection_mask = postop_mri.replace(".nii.gz", ".resection.mask.nii.gz")

            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))
            cut_coords = find_xyz_cut_coords(resection_mask)

            img = nb.load(postop_mri).get_fdata()
            pctl = np.percentile(img, 99)
            p = plot_anat(
                postop_mri,
                title=f"{subid} postop MRI with resection mask",
                axes=axes[0],
                cut_coords=cut_coords,
                vmax=pctl,
            )
            p.add_contours(resection_mask, levels=[0.5], colors="r")

            pre2post_mri = preop_mri.replace(".nii.gz", ".affine.pre2post.nii.gz")
            pre2post_mri_nonlin = preop_mri.replace(
                ".nii.gz", ".nonlin.pre2post.nii.gz"
            )

            img = nb.load(pre2post_mri).get_fdata()
            pctl = np.percentile(img, 99)
            plot_anat(
                pre2post_mri,
                title=f"{subid} pre2post MRI (affine)",
                axes=axes[1],
                cut_coords=cut_coords,
                vmax=pctl,
            )

            img = nb.load(pre2post_mri_nonlin).get_fdata()
            pctl = np.percentile(img, 99)
            plot_anat(
                pre2post_mri_nonlin,
                title=f"{subid} pre2post MRI (nonlinear)",
                axes=axes[2],
                cut_coords=cut_coords,
                vmax=pctl,
            )

            pngfile = postop_mri.replace(".nii.gz", ".resection.png")
            fig.savefig(pngfile)
            print(f"Plotted postop MRI with resection mask for {subid}")
            plt.close()
        else:
            print(f"No resection mask found for {subid}")
    else:
        print(f"Either preop or postop MRI is missing for {subid}")


def generate_preop_overlay(preop_mri, postop_mri):
    """
    Processes MRI data for a subject, generating plots of preoperative MRI with resection mask,
    affine registered postoperative MRI, and nonlinearly registered postoperative MRI.
    Parameters:
    preop_mri (str): File path to the preoperative MRI image in NIfTI format (.nii.gz).
    postop_mri (str): File path to the postoperative MRI image in NIfTI format (.nii.gz).
    The function performs the following steps:
    1. Extracts the subject ID from the preoperative MRI file name.
    2. Checks if the preoperative and postoperative MRI files exist.
    3. Checks if the resection mask file exists for the preoperative MRI.
    4. If the resection mask exists, it generates a plot with three subplots:
       - Preoperative MRI with resection mask overlay.
       - Affine registered postoperative MRI.
       - Nonlinearly registered postoperative MRI.
    5. Saves the generated plot in the specified output directory and a secondary location.
    6. Prints status messages indicating the progress and any issues encountered.
    """
    subid = os.path.basename(preop_mri).split("_")[0]

    if os.path.isfile(preop_mri) and os.path.isfile(postop_mri):
        if os.path.isfile(preop_mri.replace(".nii.gz", ".resection.mask.nii.gz")):
            resection_mask = preop_mri.replace(".nii.gz", ".resection.mask.nii.gz")

            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))
            cut_coords = find_xyz_cut_coords(resection_mask)

            img = nb.load(preop_mri).get_fdata()
            pctl = np.percentile(img, 99)
            p = plot_anat(
                preop_mri,
                title=f"{subid} preop MRI with resection mask",
                axes=axes[0],
                cut_coords=cut_coords,
                vmax=pctl,
            )
            p.add_contours(resection_mask, levels=[0.5], colors="r")

            post2pre_mri = postop_mri.replace(".nii.gz", ".affine.post2pre.nii.gz")
            post2pre_mri_nonlin = postop_mri.replace(
                ".nii.gz", ".nonlin.post2pre.nii.gz"
            )

            img = nb.load(post2pre_mri).get_fdata()
            pctl = np.percentile(img, 99)
            plot_anat(
                post2pre_mri,
                title=f"{subid} post2preop MRI (affine)",
                axes=axes[1],
                cut_coords=cut_coords,
                vmax=pctl,
            )

            img = nb.load(post2pre_mri_nonlin).get_fdata()
            pctl = np.percentile(img, 99)
            plot_anat(
                post2pre_mri_nonlin,
                title=f"{subid} post2preop MRI (nonlinear)",
                axes=axes[2],
                cut_coords=cut_coords,
                vmax=pctl,
            )

            pngfile = preop_mri.replace(".nii.gz", ".resection.png")
            fig.savefig(pngfile)
            print(f"Plotted preop MRI with resection mask for {subid}")
            plt.close()
        else:
            print(f"No resection mask found for {subid}")
    else:
        print(f"Either preop or postop MRI is missing for {subid}")


def generate_resection_overlay_plots(preop_mri, postop_mri):
    """
    Generate overlay plots for preoperative and postoperative MRI images.

    This function generates overlay plots for both preoperative and postoperative
    MRI images by calling the respective functions to create the overlays.

    Args:
        preop_mri (str): File path to the preoperative MRI image.
        postop_mri (str): File path to the postoperative MRI image.

    Returns:
        None
    """


    generate_preop_overlay(preop_mri, postop_mri)

    generate_postop_overlay(preop_mri, postop_mri)
