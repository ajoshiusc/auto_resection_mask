from genericpath import isfile
import os
import random
import nibabel as nib
from scipy import ndimage


def smooth_3d_segmentation_mask(input_path, output_path, iterations=1):
    # Read the 3D segmentation mask
    img = nib.load(input_path)
    data = img.get_fdata()

    # Apply topological dilation and erosion to smooth the segmentation boundaries
    smoothed_data = ndimage.binary_erosion(
        ndimage.binary_dilation(data, iterations=iterations), iterations=iterations
    )

    # Create a new NIfTI image with the smoothed data
    smoothed_img = nib.Nifti1Image(smoothed_data, img.affine, img.header)

    # Write the smoothed mask to the output path
    nib.save(smoothed_img, output_path)

    print(f"Smoothing completed. Smoothed mask saved to {output_path}")


# This function reads a random lesion segmentation file from the BRATS dataset
def random_lesion_segmentation(brats_data_dir):
    """
    Reads a random lesion segmentation file from the BRATS dataset.

    Parameters:
    - brats_data_dir (str): The directory path of the BRATS dataset.

    Returns:
    - random_segmentation_file (str): The file path of the randomly selected lesion segmentation file.
        Returns None if no segmentation files are found.
    """
    # Get a list of subject directories in the BRATS dataset
    subject_dirs = list(os.listdir(brats_data_dir))

    # Randomly select a subject directory
    random_subject = random.choice(subject_dirs)

    # Get the segmentation file path (assuming the segmentation file is in the "seg" subdirectory)
    random_segmentation_file = os.path.join(
        brats_data_dir, random_subject, random_subject + "_segmni.nii.gz"
    )
    # segmentation_files = [f for f in os.listdir(segmentation_dir) if f.endswith('.nii.gz')]

    if not random_segmentation_file:
        print(f"No segmentation files found for subject {random_segmentation_file}")
        return None

    return random_segmentation_file


# This function reads a random normal subject t1 file from the camcan dataset
def random_normal_subject(norm_data_dir):
    """This function reads a random normal subject t1 file from the camcan dataset.

    Args:
        norm_data_dir (str): The directory path where the normal subject data is stored.

    Returns:
        tuple: A tuple containing the file paths of the randomly selected T1 file and its corresponding mask file.
               If the T1 file or mask file is not found, None is returned.
    """

    # Get a list of subject directories in the BRATS dataset
    sub_dirs = [
        d
        for d in os.listdir(norm_data_dir)
        if os.path.isdir(os.path.join(norm_data_dir, d))
    ]

    # Randomly select a subject directory
    random_subject = random.choice(sub_dirs)
    random_subject_file = os.path.join(norm_data_dir, random_subject, "T1mni.nii.gz")
    random_subject_mask_file = random_subject_file[:-7] + ".mask.nii.gz"

    if (not os.path.isfile(random_subject_file)) or (
        not os.path.isfile(random_subject_mask_file)
    ):
        print(
            f"T1 and/or mask file(s) found {random_subject_file}, {random_subject_mask_file}"
        )
        return None

    return random_subject_file, random_subject_mask_file, random_subject
