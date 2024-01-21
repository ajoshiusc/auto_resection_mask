# AUM
# Shree Ganeshaya Namaha

import SimpleITK as sitk
import numpy as np
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from nilearn.plotting import plot_anat
from matplotlib import pyplot as plt

def find_neighboring_labels(nifti_file_path, lesion_dilated_mask, label_id):
    # Load NIfTI file
    nifti_img = nib.load(nifti_file_path)
    data = nifti_img.get_fdata()

    # Find coordinates of the given label ID
    label_coords = np.argwhere(data == label_id)

    # Get neighboring labels
    neighboring_labels = set()
    for coord in label_coords:
        x, y, z = coord
        neighbors = [
            data[x + i, y + j, z + k]
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            for k in [-1, 0, 1]
            if 0 <= x + i < data.shape[0]
            and 0 <= y + j < data.shape[1]
            and 0 <= z + k < data.shape[2]
        ]
        neighboring_labels.update(neighbors)

    # Remove the original label from the set
    neighboring_labels.discard(label_id)

    return list(neighboring_labels)


def read_nifti(file_path):
    image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(image)


def dice_coefficient(pred, truth):
    intersection = np.sum(np.logical_and(pred, truth))
    total = np.sum(pred) + np.sum(truth)

    if total == 0:
        return 1.0  # Dice is 1 if both masks are empty
    else:
        return 2.0 * intersection / total


def calculate_dice_coefficients(label_file1, label_file2, mask_file=None):

    labels1 = np.mod(read_nifti(label_file1), 1000)
    labels2 = np.mod(read_nifti(label_file2), 1000)


    dice_coefficients = []


    mask = nib.load(mask_file).get_fdata() if mask_file else None
    
    labels1 = labels1 * (mask > 0)
    labels2 = labels2 * (mask > 0)

    all_labels = np.unique(labels1[labels1>0].flatten())
    num_labels = len(all_labels)


    print("Label\tDice Coefficient")
    for label in all_labels:
        pred_label = labels1 == label
        truth_label = labels2 == label

        dice = dice_coefficient(pred_label, truth_label)
        dice_coefficients.append(dice)

        #print(f"{label}\t{dice:.4f}")

    return dice_coefficients, all_labels



neighboring_dice_coefficients_avg=[]
neighboring_dice_coefficients_avg2=[]

for subno in range(29,32):

    if subno == 30:
        continue

    dilation_distance_mm = 5.0
    # Replace 'label_file1.nii' and 'label_file2.nii' with your actual file paths
    label_ground_truth = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_warp.withlesion.label.nii.gz"
    label_lesion_brain = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_orig_BrainSuite/Subject_{subno}_orig.svreg.label.nii.gz"
    label_inpained_moved = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_moved_labels.nii.gz"

    lesion_mask = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_lesion_mask.nii.gz"

    lesion_neighborhood_file = lesion_mask.replace('.nii.gz',f'_neighborhood_{dilation_distance_mm}.nii.gz')

    dice_coefficients_neighborhood, all_labels = calculate_dice_coefficients(label_ground_truth, label_lesion_brain, lesion_neighborhood_file)
    dice_coefficients_neighborhood2, all_labels_neighborhood2 = calculate_dice_coefficients(label_ground_truth, label_inpained_moved, lesion_neighborhood_file)

    
    all_labels = list(all_labels)
   
    # find average dice coefficients for overlapping and neighboring labels
    neighboring_dice_coefficients_avg.append(np.mean(dice_coefficients_neighborhood))

    neighboring_dice_coefficients_avg2.append(np.mean(dice_coefficients_neighborhood2))

    print('**********************************************************************************************')


    # print a table of dice coefficients for each subject
    print("-------\t-----------\t-----------")
    for i in range(len(dice_coefficients_neighborhood)):
        print(f"{i}\t{dice_coefficients_neighborhood[i]:.4f}\t{dice_coefficients_neighborhood2[i]:.4f}")

    print('**********************************************************************************************')

 

print(f"Average Dice Coefficient for neighboring labels: {np.mean(neighboring_dice_coefficients_avg):.4f}")
print(f"Average Dice Coefficient for neighboring labels: {np.mean(neighboring_dice_coefficients_avg2):.4f}")

print('**********************************************************************************************')

# print a table of dice coefficients for each subject
print("-------\t-----------\t-----------")
for i in range(len(neighboring_dice_coefficients_avg)):
    print(f"{i}\t{neighboring_dice_coefficients_avg[i]:.4f}\t{neighboring_dice_coefficients_avg2[i]:.4f}")

print('**********************************************************************************************')

 




