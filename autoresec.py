import nilearn.image as ni
from aligner import Aligner, center_and_resample_images
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
)
import nibabel as nib
import os
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, opening,closing, erosion, isotropic_erosion, dilation
from warper import Warper
import nibabel.processing as nibp
from skimage.measure import label
import tempfile
from shutil import copyfile
from warp_utils import apply_warp

from nilearn import image

def apply_mask(input_file, mask_file, output_file):
    # Load input image and mask
    img = image.load_img(input_file)
    mask = image.load_img(mask_file)

    # Apply mask to image data
    masked_img = image.math_img("img * mask", img=img, mask=mask)

    # Save masked image to output file
    masked_img.to_filename(output_file)
    print(f"Mask applied successfully. Saved to: {output_file}")



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

        center_and_resample_images(
            pre_mri_path,
            bst_atlas_path,
            centered_atlas=cent_bst_atlas,
            atlas_labels=bst_atlas_labels_path,
            centered_atlas_labels=cent_bst_atlas_labels,
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

        # plot_roi(temp_out_labels, bg_img=pre_mri_path, title="out-labels over pre-mri")
        # plt.show()

        labels_to_zero = [3, 4, 5, 6]

        # Load the NIfTI file
        img = nib.load(temp_out_labels)

        # Get the data as a NumPy array
        data = img.get_fdata()

        # Set specific labels to zero
        for label_id in labels_to_zero:
            data[data == label_id] = 0

        data[data > 0] = 255

        # erode mask by 3 voxel
        data = isotropic_erosion(data, 3) # (data, np.ones((9, 9, 9)))

        # Create a new NIfTI image with modified data
        modified_img = nib.Nifti1Image(np.uint8(data), img.affine)

        # Save the modified NIfTI image to a new file
        nib.save(modified_img, output_possible_resect_mask_path)


def delineate_resection_pre(
    pre_mri_path,
    post_mri_path,
    BrainSuitePATH="/home/ajoshi/Software/BrainSuite23a",
    ERR_THR=80,
    bst_atlas_path="/deneb_disk/auto_resection/bst_atlases/icbm_bst.nii.gz",
    bst_atlas_labels_path="/deneb_disk/auto_resection/bst_atlases/icbm_bst.label.nii.gz",
):
    # pl.plot_anat(pre_mri,title='pre-mri')
    # pl.plot_anat(post_mri,title='post-mri')

    # pl.show()

    # %%

    affine_reg = Aligner()

    pre_mri_base_orig = pre_mri_path[:-7]
    post_mri_base_orig = post_mri_path[:-7]

    pth_pre, base_pre = os.path.split(pre_mri_base_orig)
    pth_post, base_post = os.path.split(post_mri_base_orig)

    # %% [markdown]
    # Create a temporary directory where intermediate outputs will be stored. Currently it is not deleted, but you can delete this directory later on if you want.
    #
    # The images are resampled to 1mm cubic voxels

    # %%

    temp_pth = os.path.join(pth_pre, "temp_dir_resection")

    print(f"Temporary files will be stored in {temp_pth}")

    if not os.path.isdir(temp_pth):
        os.makedirs(temp_pth)

    pre_mri_base = os.path.join(temp_pth, base_pre + "_1mm")
    post_mri_base = os.path.join(temp_pth, base_post + "_1mm")

    out_img = nibp.conform(nib.load(pre_mri_base_orig + ".nii.gz"))
    out_img.to_filename(pre_mri_base + ".nii.gz")

    out_img = nibp.conform(nib.load(post_mri_base_orig + ".nii.gz"))
    out_img.to_filename(post_mri_base + ".nii.gz")

    pre_mri_dir, _ = os.path.split(pre_mri_base)

    mov_img_orig = post_mri_base + ".nii.gz"

    # If the nii.gz file does not exist then maybe .nii file exist. Check your input

    if not os.path.isfile(mov_img_orig):
        mov_img_orig = post_mri_base + ".nii"

    # %%

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "bse")
        + " -i "
        + pre_mri_base
        + ".nii.gz"
        + " -o "
        + pre_mri_base
        + ".bse.nii.gz"
        + " --auto --trim --mask "
        + pre_mri_base
        + ".mask.nii.gz"
    )
    os.system(cmd)

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "bfc")
        + " -i "
        + pre_mri_base
        + ".bse.nii.gz"
        + " -o "
        + pre_mri_base
        + ".bfc.nii.gz"
        + " -m "
        + pre_mri_base
        + ".mask.nii.gz"
    )
    os.system(cmd)

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "pvc")
        + " -i "
        + pre_mri_base
        + ".bfc.nii.gz"
        + " -o "
        + pre_mri_base
        + ".pvc.label.nii.gz"
        + " -f "
        + pre_mri_base
        + ".pvc.frac.nii.gz"
    )
    os.system(cmd)

    # Post MRI pre processing

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "bse")
        + " -i "
        + post_mri_base
        + ".nii.gz"
        + " -o "
        + post_mri_base
        + ".bse.nii.gz"
        + " --auto --trim --mask "
        + post_mri_base
        + ".mask.nii.gz"
    )
    os.system(cmd)

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "bfc")
        + " -i "
        + post_mri_base
        + ".bse.nii.gz"
        + " -o "
        + post_mri_base
        + ".bfc.nii.gz"
        + " -m "
        + post_mri_base
        + ".mask.nii.gz"
    )
    os.system(cmd)

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "pvc")
        + " -i "
        + post_mri_base
        + ".bfc.nii.gz"
        + " -o "
        + post_mri_base
        + ".pvc.label.nii.gz"
        + " -f "
        + post_mri_base
        + ".pvc.frac.nii.gz"
    )
    os.system(cmd)

    # %% [markdown]
    # # Do Affine Registration of post-op MRI to pre-op MRI
    #
    # The following code uses CNN to register pos-top mri to pre-op mri

    # %%

    mov_img = post_mri_base + ".pvc.frac.nii.gz"
    ref_img = pre_mri_base + ".pvc.frac.nii.gz"
    ref_img_mask = pre_mri_base + ".mask.nii.gz"
    mov_img_mask = post_mri_base + ".mask.nii.gz"

    ref_img_pvc_frac = pre_mri_base + ".pvc.frac.nii.gz"
    error_img = pre_mri_dir + "/error_pre_post.nii.gz"
    error_mask_img = pre_mri_dir + "/error_pre_post.mask.nii.gz"
    error_init_mask_img = pre_mri_dir + "/error_pre_post.init.mask.nii.gz"

    # rigidly warped image
    affine_reg_img = pre_mri_dir + "/post2pre.nii.gz"
    affine_reg_img_bse = pre_mri_dir + "/post2pre.bse.nii.gz"
    affine_reg_img_mask = pre_mri_dir + "/post2pre.mask.nii.gz"
    affine_reg_img_bfc = pre_mri_dir + "/post2pre.bfc.nii.gz"
    affine_reg_img_pvc_label = pre_mri_dir + "/post2pre.pvc.label.nii.gz"
    affine_reg_img_pvc_frac = pre_mri_dir + "/post2pre.pvc.frac.nii.gz"

    nonlin_reg_img = pre_mri_dir + "/post2pre.nonlin.nii.gz"

    ddf = pre_mri_dir + "/ddf.nii.gz"

    affine_reg.affine_reg(
        fixed_file=ref_img,
        moving_file=mov_img,
        output_file=affine_reg_img_bse,
        ddf_file=ddf,
        loss="cc",
        nn_input_size=64,
        lr=1e-6,
        max_epochs=1500,
        device="cuda",
    )

    moving = LoadImage(image_only=True)(mov_img_orig)
    moving = EnsureChannelFirst()(moving)

    target = LoadImage(image_only=True)(ref_img)
    target = EnsureChannelFirst()(target)

    image_movedo = apply_warp(affine_reg.ddf[None,], moving[None,], target[None,])

    nib.save(
        nib.Nifti1Image(
            image_movedo[0, 0].detach().cpu().numpy(), affine_reg.target.affine
        ),
        affine_reg_img,
    )

    

    # plotting.plot_anat(affine_reg_img, title="affine registered post-mri",cut_coords=(0,0,0))
    # plotting.plot_anat(pre_mri, title="Pre MRI",cut_coords=(0,0,0))
    # plotting.show()

    # %%


    print('use common mask for pre and post mri')

    

    #cmd = (
    #    os.path.join(BrainSuitePATH, "bin", "bse")
    #    + " -i "
    #    + affine_reg_img
    #    + " -o "
    #    + affine_reg_img_bse
    #    + " --auto --trim --mask "
    #    + affine_reg_img_mask
    #)
    #os.system(cmd)

    # apply mask from pre-mri to affine_reg_img

    #copyfile(ref_img_mask,affine_reg_img_mask)

    copyfile(pre_mri_base + ".mask.nii.gz",affine_reg_img_mask)

    # apply affine_reg_img_mask to affine_reg_img using nilearn
    apply_mask(affine_reg_img, affine_reg_img_mask, affine_reg_img_bse)

    # bfc processing
    cmd = (
        os.path.join(BrainSuitePATH, "bin", "bfc")
        + " -i "
        + affine_reg_img_bse
        + " -o "
        + affine_reg_img_bfc
        + " -m "
        + affine_reg_img_mask
    )
    os.system(cmd)

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "pvc")
        + " -i "
        + affine_reg_img_bfc
        + " -o "
        + affine_reg_img_pvc_label
        + " -f "
        + affine_reg_img_pvc_frac
    )
    os.system(cmd)

    # %% [markdown]
    # Apply Afiine registrations to the images and find initial estimate of don't care region

    # %%

    # Load the images and normalize their intensities
    vref = LoadImage(image_only=True)(ref_img_pvc_frac)
    vwrp = LoadImage(image_only=True)(affine_reg_img_pvc_frac)
    msk = LoadImage(image_only=True)(ref_img_mask)

    msk = (msk>0) & (vwrp>0)
    vwrp = (255.0 / np.max(vwrp[msk > 0])) * vwrp
    vref = (255.0 / np.max(vref[msk > 0])) * vref

    # compute the error and smooth the error
    #vwrp = np.sqrt((vref - vwrp) ** 2)
    vwrp = vref - vwrp

    vwrp = vwrp * (msk > 0)
    vwrp = gaussian_filter(vwrp, sigma=1)

    nib.save(nib.Nifti1Image(vwrp, affine_reg.target.affine), error_img)

    error_mask = opening((vwrp > ERR_THR) | (vwrp < -ERR_THR))
    nib.save(
        nib.Nifti1Image(255 * np.uint8(error_mask), affine_reg.target.affine),
        error_init_mask_img,
    )

    resection_mask = remove_small_objects(error_mask)
    nib.save(
        nib.Nifti1Image(255 * np.uint8(resection_mask), affine_reg.target.affine),
        error_mask_img,
    )

    # %%
    # plotting.plot_roi(roi_img=error_mask_img, bg_img=ref_img_pvc_frac, title="initial estimate of resected area")

    # %%

    nonlin_reg = Warper()

    ref_img_bfc = pre_mri_base + ".bfc.nii.gz"

    error_img = pre_mri_dir + "/error_pre_post.nonlin.nii.gz"

    error_init_mask_img = pre_mri_dir + "/error_pre_post.nonlin.init.mask.nii.gz"

    error_mask_img_nonlin = pre_mri_dir + "/error_pre_post.nonlin.mask.nii.gz"

    error_mask_img_rigid = pre_mri_dir + "/error_pre_post.init.mask.nii.gz"

    target_msk_file = pre_mri_dir + "/target.mask.nii.gz"

    # rigidly warped image

    nonlin_reg_img_pvc_frac = pre_mri_dir + "/post2pre.nonlin.warped.pvc.frac.nii.gz"

    jac_file = pre_mri_dir + "/jacobian.nii.gz"

    ddf = pre_mri_dir + "/ddf_nonlin.nii.gz"

    possible_resec_mask = pre_mri_dir + "/possible_resec_mask.nii.gz"

    tar_msk, target_mask_meta = LoadImage(image_only=False)(error_mask_img_rigid)
    tar_msk = gaussian_filter(tar_msk, sigma=1)
    tar_msk = np.float32(tar_msk < 1)

    nib.save(
        nib.Nifti1Image(255 * (tar_msk), target_mask_meta["affine"]),
        target_msk_file,
    )

    # %% [markdown]
    # # Do Non-linear transformation
    # The following code uses CNN to do nonlinear registration

    # %%

    nonlin_reg.nonlinear_reg(
        target_file=ref_img_pvc_frac,
        moving_file=affine_reg_img_pvc_frac,
        output_file=nonlin_reg_img_pvc_frac,
        ddf_file=ddf,
        reg_penalty=.01,
        nn_input_size=64,
        lr=1e-3,
        max_epochs=1000,
        loss="mse",
        jacobian_determinant_file=jac_file,
        target_mask=target_msk_file,
    )

    # %%

    vref = LoadImage(image_only=True)(ref_img_pvc_frac)
    vwrp = LoadImage(image_only=True)(nonlin_reg_img_pvc_frac)
    msk = LoadImage(image_only=True)(ref_img_mask)

    print(ref_img_pvc_frac)
    print(nonlin_reg_img_pvc_frac)
    print(ref_img_mask)

    # %%

    def get_largest_cc(segmentation):
        labels = label(segmentation)
        assert labels.max() != 0  # assume at least 1 CC
        largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largest_cc

    # %%

    msk = (msk>0) & (vwrp>0)

    vwrp = vref - vwrp

    nib.save(
        nib.Nifti1Image(vwrp.detach().numpy(), nonlin_reg.target.affine), error_img
    )
    vwrp = vwrp * (msk > 0)

    ST = 3
    ERR_THR = 0.75 #(ERR_THR*3.0)/255.0 #0.99
    #error_mask = opening(opening(closing(dilation(np.array(vwrp)))) > ERR_THR, footprint=[(np.ones((ST, 1, 1)), 1), (np.ones((1, ST, 1)), 1), (np.ones((1, 1, ST)), 1),],)
    error_mask_core = opening(vwrp > ERR_THR, footprint=[(np.ones((ST, 1, 1)), 1), (np.ones((1, ST, 1)), 1), (np.ones((1, 1, ST)), 1)])
    
    error_mask_dilated = dilation(error_mask_core, np.ones((7, 7, 7)))

    error_mask = opening(closing((dilation(vwrp) > ERR_THR))*(error_mask_dilated>0))

    nib.save(
        nib.Nifti1Image(255 * np.uint8(error_mask), nonlin_reg.target.affine),
        error_init_mask_img,
    )

    get_possible_resect_mask(
        nonlin_reg_img_pvc_frac,
        possible_resec_mask,
        bst_atlas_path,
        bst_atlas_labels_path,
    )

    msk = nib.load(possible_resec_mask).get_fdata()
    resection_mask = error_mask * (msk > 0)

    #resection_mask = remove_small_objects(resection_mask)
    resection_mask = get_largest_cc(resection_mask)

    nib.save(
        nib.Nifti1Image(255 * np.uint8(resection_mask), nonlin_reg.target.affine),
        error_mask_img_nonlin,
    )

    # %% [markdown]
    # Write Mask to pre_image space

    # %%
    output_mask_pre = pre_mri_base_orig + ".resection.mask.nii.gz"
    output_mask_post = post_mri_base_orig + ".resection.mask.nii.gz"

    affine_reg_img_out = post_mri_base_orig + ".affine.post2pre.nii.gz"
    nonlin_reg_img_out = post_mri_base_orig + ".nonlin.post2pre.nii.gz"

    # apply nonlinear warp to post-mri
    moving = LoadImage(image_only=True)(affine_reg_img)
    moving = EnsureChannelFirst()(moving)

    target = LoadImage(image_only=True)(ref_img)
    target = EnsureChannelFirst()(target)

    image_movedo = apply_warp(nonlin_reg.ddf[None,], moving[None,], target[None,])

    nib.save(
        nib.Nifti1Image(
            image_movedo[0, 0].detach().cpu().numpy(), affine_reg.target.affine
        ),
        nonlin_reg_img,
    )




    # copyfile(affine_reg_img, affine_reg_img_out)

    ni.resample_to_img(
        affine_reg_img, pre_mri_path, interpolation="linear",force_resample=True, copy_header=True
    ).to_filename(affine_reg_img_out)

    ni.resample_to_img(
        nonlin_reg_img, pre_mri_path, interpolation="linear",force_resample=True, copy_header=True
    ).to_filename(nonlin_reg_img_out)

    ni.resample_to_img(
        error_mask_img_nonlin, pre_mri_path, interpolation="nearest", force_resample=True, copy_header=True
    ).to_filename(output_mask_pre)

    # %%
    # plotting.plot_roi(output_mask_pre,bg_img=pre_mri,cmap='Wistia')

    return error_mask_img_nonlin






def delineate_resection_post(
    pre_mri_path,
    post_mri_path,
    BrainSuitePATH="/home/ajoshi/Software/BrainSuite23a",
    ERR_THR=80,
    bst_atlas_path="/deneb_disk/auto_resection/bst_atlases/icbm_bst.nii.gz",
    bst_atlas_labels_path="/deneb_disk/auto_resection/bst_atlases/icbm_bst.label.nii.gz",
):
    # pl.plot_anat(pre_mri,title='pre-mri')
    # pl.plot_anat(post_mri,title='post-mri')

    # pl.show()

    # %%

    affine_reg = Aligner()

    pre_mri_base_orig = pre_mri_path[:-7]
    post_mri_base_orig = post_mri_path[:-7]

    _, base_pre = os.path.split(pre_mri_base_orig)
    pth_post, base_post = os.path.split(post_mri_base_orig)

    # %% [markdown]
    # Create a temporary directory where intermediate outputs will be stored. Currently it is not deleted, but you can delete this directory later on if you want.
    #
    # The images are resampled to 1mm cubic voxels

    # %%

    temp_pth = os.path.join(pth_post, "temp_dir_resection")

    print(f"Temporary files will be stored in {temp_pth}")

    if not os.path.isdir(temp_pth):
        os.makedirs(temp_pth)

    pre_mri_base = os.path.join(temp_pth, base_pre + "_1mm")
    post_mri_base = os.path.join(temp_pth, base_post + "_1mm")

    out_img = nibp.conform(nib.load(pre_mri_base_orig + ".nii.gz"))
    out_img.to_filename(pre_mri_base + ".nii.gz")

    out_img = nibp.conform(nib.load(post_mri_base_orig + ".nii.gz"))
    out_img.to_filename(post_mri_base + ".nii.gz")

    pre_mri_dir, _ = os.path.split(pre_mri_base)
    post_mri_dir, _ = os.path.split(post_mri_base)

    mov_img_orig = pre_mri_base + ".nii.gz"

    # If the nii.gz file does not exist then maybe .nii file exist. Check your input

    if not os.path.isfile(mov_img_orig):
        mov_img_orig = pre_mri_base + ".nii"

    # %%

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "bse")
        + " -i "
        + pre_mri_base
        + ".nii.gz"
        + " -o "
        + pre_mri_base
        + ".bse.nii.gz"
        + " --auto --trim --mask "
        + pre_mri_base
        + ".mask.nii.gz"
    )
    os.system(cmd)

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "bfc")
        + " -i "
        + pre_mri_base
        + ".bse.nii.gz"
        + " -o "
        + pre_mri_base
        + ".bfc.nii.gz"
        + " -m "
        + pre_mri_base
        + ".mask.nii.gz"
    )
    os.system(cmd)

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "pvc")
        + " -i "
        + pre_mri_base
        + ".bfc.nii.gz"
        + " -o "
        + pre_mri_base
        + ".pvc.label.nii.gz"
        + " -f "
        + pre_mri_base
        + ".pvc.frac.nii.gz"
    )
    os.system(cmd)

    # Post MRI pre processing

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "bse")
        + " -i "
        + post_mri_base
        + ".nii.gz"
        + " -o "
        + post_mri_base
        + ".bse.nii.gz"
        + " --auto --trim --mask "
        + post_mri_base
        + ".mask.nii.gz"
    )
    os.system(cmd)

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "bfc")
        + " -i "
        + post_mri_base
        + ".bse.nii.gz"
        + " -o "
        + post_mri_base
        + ".bfc.nii.gz"
        + " -m "
        + post_mri_base
        + ".mask.nii.gz"
    )
    os.system(cmd)

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "pvc")
        + " -i "
        + post_mri_base
        + ".bfc.nii.gz"
        + " -o "
        + post_mri_base
        + ".pvc.label.nii.gz"
        + " -f "
        + post_mri_base
        + ".pvc.frac.nii.gz"
    )
    os.system(cmd)

    # %% [markdown]
    # # Do Affine Registration of post-op MRI to pre-op MRI
    #
    # The following code uses CNN to register pos-top mri to pre-op mri

    # %%

    mov_img = pre_mri_base + ".pvc.frac.nii.gz"
    ref_img = post_mri_base + ".pvc.frac.nii.gz"
    ref_img_mask = post_mri_base + ".mask.nii.gz"
    ref_img_pvc_frac = post_mri_base + ".pvc.frac.nii.gz"
    error_img = post_mri_dir + "/error_post_pre.nii.gz"
    error_mask_img = post_mri_dir + "/error_post_pre.mask.nii.gz"
    error_init_mask_img = post_mri_dir + "/error_post_pre.init.mask.nii.gz"

    # rigidly warped image
    affine_reg_img = post_mri_dir + "/pre2post.nii.gz"
    affine_reg_img_bse = post_mri_dir + "/pre2post.bse.nii.gz"
    affine_reg_img_mask = post_mri_dir + "/pre2post.mask.nii.gz"
    affine_reg_img_bfc = post_mri_dir + "/pre2post.bfc.nii.gz"
    affine_reg_img_pvc_label = post_mri_dir + "/pre2post.pvc.label.nii.gz"
    affine_reg_img_pvc_frac = post_mri_dir + "/pre2post.pvc.frac.nii.gz"

    nonlin_reg_img = post_mri_dir + "/post2pre.nonlin.nii.gz"

    ddf = pre_mri_dir + "/ddf.nii.gz"

    affine_reg.affine_reg(
        fixed_file=ref_img,
        moving_file=mov_img,
        output_file=affine_reg_img_bse,
        ddf_file=ddf,
        loss="cc",
        nn_input_size=64,
        lr=1e-6,
        max_epochs=1500,
        device="cuda",
    )

    moving = LoadImage(image_only=True)(mov_img_orig)
    moving = EnsureChannelFirst()(moving)

    target = LoadImage(image_only=True)(ref_img)
    target = EnsureChannelFirst()(target)

    image_movedo = apply_warp(affine_reg.ddf[None,], moving[None,], target[None,])

    nib.save(
        nib.Nifti1Image(
            image_movedo[0, 0].detach().cpu().numpy(), affine_reg.target.affine
        ),
        affine_reg_img,
    )

    # plotting.plot_anat(affine_reg_img, title="affine registered post-mri",cut_coords=(0,0,0))
    # plotting.plot_anat(pre_mri, title="Pre MRI",cut_coords=(0,0,0))
    # plotting.show()

    # %%

    # cmd = (
    #     os.path.join(BrainSuitePATH, "bin", "bse")
    #     + " -i "
    #     + affine_reg_img
    #     + " -o "
    #     + affine_reg_img_bse
    #     + " --auto --trim --mask "
    #     + affine_reg_img_mask
    # )
    # os.system(cmd)

    # copy mask
    copyfile(ref_img_mask,affine_reg_img_mask)

    # apply affine_reg_img_mask to affine_reg_img using nilearn
    apply_mask(affine_reg_img, affine_reg_img_mask, affine_reg_img_bse)

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "bfc")
        + " -i "
        + affine_reg_img_bse
        + " -o "
        + affine_reg_img_bfc
        + " -m "
        + affine_reg_img_mask
    )
    os.system(cmd)

    cmd = (
        os.path.join(BrainSuitePATH, "bin", "pvc")
        + " -i "
        + affine_reg_img_bfc
        + " -o "
        + affine_reg_img_pvc_label
        + " -f "
        + affine_reg_img_pvc_frac
    )
    os.system(cmd)

    # %% [markdown]
    # Apply Afiine registrations to the images and find initial estimate of don't care region

    # %%

    # Load the images and normalize their intensities
    vref = LoadImage(image_only=True)(ref_img_pvc_frac)
    vwrp = LoadImage(image_only=True)(affine_reg_img_pvc_frac)
    msk = LoadImage(image_only=True)(ref_img_mask)

    vwrp = (255.0 / np.max(vwrp[msk > 0])) * vwrp
    vref = (255.0 / np.max(vref[msk > 0])) * vref

    # compute the error and smooth the error
    #vwrp = np.sqrt((vref - vwrp) ** 2)
    vwrp = -(vref - vwrp)

    vwrp = vwrp * (msk > 0)
    vwrp = gaussian_filter(vwrp, sigma=1)

    nib.save(nib.Nifti1Image(vwrp, affine_reg.target.affine), error_img)

    error_mask = opening((vwrp > ERR_THR) | (vwrp < -ERR_THR), footprint=[(np.ones((3, 1, 1)), 1), (np.ones((1, 3, 1)), 1), (np.ones((1, 1, 3)), 1)]) 
    nib.save(
        nib.Nifti1Image(255 * np.uint8(error_mask), affine_reg.target.affine),
        error_init_mask_img,
    )

    resection_mask = remove_small_objects(error_mask)
    nib.save(
        nib.Nifti1Image(255 * np.uint8(resection_mask), affine_reg.target.affine),
        error_mask_img,
    )

    # %%
    # plotting.plot_roi(roi_img=error_mask_img, bg_img=ref_img_pvc_frac, title="initial estimate of resected area")

    # %%

    nonlin_reg = Warper()

    ref_img_bfc = post_mri_base + ".bfc.nii.gz"

    error_img = post_mri_dir + "/error_post_pre.nonlin.nii.gz"

    error_init_mask_img = post_mri_dir + "/error_post_pre.nonlin.init.mask.nii.gz"

    error_mask_img_nonlin = post_mri_dir + "/error_post_pre.nonlin.mask.nii.gz"

    error_mask_img_rigid = post_mri_dir + "/error_post_pre.init.mask.nii.gz"

    target_msk_file = post_mri_dir + "/target.mask.nii.gz"

    # rigidly warped image

    nonlin_reg_img_pvc_frac = post_mri_dir + "/pre2post.nonlin.warped.pvc.frac.nii.gz"

    jac_file = post_mri_dir + "/jacobian.nii.gz"

    ddf = post_mri_dir + "/ddf_nonlin.nii.gz"

    possible_resec_mask = post_mri_dir + "/possible_resec_post_mask.nii.gz"

    tar_msk, target_mask_meta = LoadImage(image_only=False)(error_mask_img_rigid)
    tar_msk = gaussian_filter(tar_msk, sigma=1)
    tar_msk = np.float32(tar_msk < 1)

    nib.save(
        nib.Nifti1Image(255 * (tar_msk), target_mask_meta["affine"]),
        target_msk_file,
    )

    # %% [markdown]
    # # Do Non-linear transformation
    # The following code uses CNN to do nonlinear registration

    # %%

    nonlin_reg.nonlinear_reg(
        target_file=ref_img_pvc_frac,
        moving_file=affine_reg_img_pvc_frac,
        output_file=nonlin_reg_img_pvc_frac,
        ddf_file=ddf,
        reg_penalty=3,
        nn_input_size=64,
        lr=1e-3,
        max_epochs=1000,
        loss="mse",
        jacobian_determinant_file=jac_file,
        target_mask=target_msk_file,
    )

    # %%

    vref = LoadImage(image_only=True)(ref_img_pvc_frac)
    vwrp = LoadImage(image_only=True)(nonlin_reg_img_pvc_frac)
    msk = LoadImage(image_only=True)(ref_img_mask)

    print(ref_img_pvc_frac)
    print(nonlin_reg_img_pvc_frac)
    print(ref_img_mask)

    # %%

    def get_largest_cc(segmentation):

        if np.sum(segmentation) == 0:
            return segmentation
        else:
            labels = label(segmentation)
            #assert labels.max() != 0  # assume at least 1 CC
            print(f' size of largest connected component is: {labels.max()}')
            largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
            return largest_cc

    # %%

    vwrp = -(vref - vwrp)

    nib.save(
        nib.Nifti1Image(vwrp.detach().numpy(), nonlin_reg.target.affine), error_img
    )
    vwrp = vwrp * (msk > 0)

    ST = 3
    ERR_THR = (ERR_THR*3.0)/255.0 #0.99
    error_mask = opening(
        (vwrp > ERR_THR) | (vwrp < -ERR_THR),
        footprint=[
            (np.ones((ST, 1, 1)), 1),
            (np.ones((1, ST, 1)), 1),
            (np.ones((1, 1, ST)), 1),
        ],
    )
    nib.save(
        nib.Nifti1Image(255 * np.uint8(error_mask), nonlin_reg.target.affine),
        error_init_mask_img,
    )

    get_possible_resect_mask(
        nonlin_reg_img_pvc_frac,
        possible_resec_mask,
        bst_atlas_path,
        bst_atlas_labels_path,
    )

    msk = nib.load(possible_resec_mask).get_fdata()
    resection_mask = error_mask * (msk > 0)

    
    #resection_mask = remove_small_objects(resection_mask)
    resection_mask = get_largest_cc(resection_mask)

    nib.save(
        nib.Nifti1Image(255 * np.uint8(resection_mask), nonlin_reg.target.affine),
        error_mask_img_nonlin,
    )

    # %% [markdown]
    # Write Mask to pre_image space


    # apply nonlinear warp to post-mri
    moving = LoadImage(image_only=True)(affine_reg_img)
    moving = EnsureChannelFirst()(moving)

    target = LoadImage(image_only=True)(ref_img)
    target = EnsureChannelFirst()(target)

    image_movedo = apply_warp(nonlin_reg.ddf[None,], moving[None,], target[None,])

    nib.save(
        nib.Nifti1Image(
            image_movedo[0, 0].detach().cpu().numpy(), affine_reg.target.affine
        ),
        nonlin_reg_img,
    )



    # %%




    output_mask_pre = pre_mri_base_orig + ".resection.mask.nii.gz"
    output_mask_post = post_mri_base_orig + ".resection.mask.nii.gz"

    affine_reg_img_out = pre_mri_base_orig + ".affine.pre2post.nii.gz"
    nonlin_reg_img_out = pre_mri_base_orig + ".nonlin.pre2post.nii.gz"

    # copyfile(affine_reg_img, affine_reg_img_out)

    ni.resample_to_img(
        affine_reg_img, post_mri_path, interpolation="linear", force_resample=True, copy_header=True
    ).to_filename(affine_reg_img_out)

    ni.resample_to_img(
        nonlin_reg_img, post_mri_path, interpolation="linear",force_resample=True, copy_header=True
    ).to_filename(nonlin_reg_img_out)


    ni.resample_to_img(
        error_mask_img_nonlin, post_mri_path, interpolation="nearest", force_resample=True, copy_header=True
    ).to_filename(output_mask_post)

    # %%
    # plotting.plot_roi(output_mask_pre,bg_img=pre_mri,cmap='Wistia')

    return error_mask_img_nonlin






if __name__ == "__main__":

    print(f'is cuda available? {torch.cuda.is_available()}')

    # Example usage:
    pre_mri = (
        "/deneb_disk/auto_resection/test/sub-0003/preop/sub-0003_preop-t1mri-1.nii.gz"
    )
    post_mri = (
        "/deneb_disk/auto_resection/test/sub-0003/postop/sub-0003_postop-t1mri-1.nii.gz"
    )

    output_resection_mask = delineate_resection(pre_mri, post_mri)
    print("Resection mask saved to:", output_resection_mask)

    output_resection_mask = delineate_resection_post(pre_mri, post_mri)
    print("Resection mask saved to:", output_resection_mask)
