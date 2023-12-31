from genericpath import isfile
import os
import random
import nibabel as nib
from scipy import ndimage
import glob

import csv
import vtk
from pathlib import Path
import numpy as np

def read_poly_data(path, flip=False):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly_data = reader.GetOutput()
    if flip:
        from .mesh import flipxy
        poly_data = flipxy(poly_data)
    return poly_data

def get_sphere_poly_data():
    resources_dir = Path(__file__).parent / 'resources'
    mesh_path = resources_dir / 'geodesic_polyhedron.vtp'
    if not mesh_path.is_file():
        raise FileNotFoundError(f'{mesh_path} does not exist')
    poly_data = read_poly_data(mesh_path)
    if poly_data.GetNumberOfPoints() == 0:
        message = (
            f'Error reading sphere poly data from {mesh_path}. Contents:'
            f'\n{mesh_path.read_text()}'
        )
        raise FileNotFoundError(message)
    return poly_data



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

    random_t1_file = os.path.join(
        brats_data_dir, random_subject, random_subject + "_t1mni.nii.gz"
    )
    # segmentation_files = [f for f in os.listdir(segmentation_dir) if f.endswith('.nii.gz')]

    if not random_segmentation_file:
        print(f"No segmentation files found for subject {random_segmentation_file}")
        return None

    return random_segmentation_file, random_t1_file


# This function reads a random lesion segmentation file from the BRATS dataset
def random_lesion_segmentation_carc(brats_data_dir):
    """
    Reads a random lesion segmentation file from the BRATS dataset.

    Parameters:
    - brats_data_dir (str): The directory path of the BRATS dataset.

    Returns:
    - random_segmentation_file (str): The file path of the randomly selected lesion segmentation file.
        Returns None if no segmentation files are found.
    """
    # Get a list of subject directories in the BRATS dataset
    subject_t1_files = glob.glob(brats_data_dir+'/t1/B*.nii.gz')

    # Randomly select a subject directory
    random_t1_file = random.choice(subject_t1_files)
    _, t1_fname = os.path.split(random_t1_file)

    random_subject = t1_fname[:-10]
    # Get the segmentation file path (assuming the segmentation file is in the "seg" subdirectory)
    random_segmentation_file = os.path.join(
        brats_data_dir, 'seg',random_subject + "_seg.nii.gz"
    )

    # segmentation_files = [f for f in os.listdir(segmentation_dir) if f.endswith('.nii.gz')]

    if (not os.path.isfile(random_segmentation_file)) or (not os.path.isfile(random_t1_file)) :
        print(f"No segmentation files found for subject {random_segmentation_file}")
        return None

    return random_segmentation_file, random_t1_file

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



# This function reads a random normal subject t1 file from the camcan dataset
def random_normal_subject_carc(norm_data_dir):
    """This function reads a random normal subject t1 file from the camcan dataset.

    Args:
        norm_data_dir (str): The directory path where the normal subject data is stored.

    Returns:
        tuple: A tuple containing the file paths of the randomly selected T1 file and its corresponding mask file.
               If the T1 file or mask file is not found, None is returned.
    """

    # Get a list of subject directories in the BRATS dataset
    
    sub_list = []

    with open('/project/ajoshi_27/akrami/3D_lesion_DF/Data/splits/IXI_test.csv', 'r') as rf:
        
        reader = csv.reader(rf, delimiter=',')
        next(reader, None) # ignore the header
        for row in reader:
            sub_list.append(row[1])
            print(row[1])


    #sub_list = glob.glob(norm_data_dir+'/IXI*_t1.nii.gz')

    # Randomly select a subject directory
    random_subject_file = os.path.join(norm_data_dir, 'Train/ixi/t1', random.choice(sub_list))

    if (not os.path.isfile(random_subject_file)):
        print(
            f"T1 and/or mask file(s) found {random_subject_file}"
        )
        return None

    _,base = os.path.split(random_subject_file)
    random_subject = base[:-10]

    return random_subject_file, random_subject
