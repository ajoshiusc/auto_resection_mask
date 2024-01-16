# AUM
# Shree Ganeshaya Namaha

import SimpleITK as sitk
import numpy as np
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from nilearn.plotting import plot_anat
from matplotlib import pyplot as plt

def find_neighboring_labels(nifti_file_path, label_id):
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


def calculate_dice_coefficients(label_file1, label_file2):
    labels1 = np.mod(read_nifti(label_file1), 1000)
    labels2 = np.mod(read_nifti(label_file2), 1000)

    all_labels = np.unique(labels1.flatten())
    num_labels = len(all_labels)

    dice_coefficients = []

    print("Label\tDice Coefficient")
    for label in all_labels:
        pred_label = labels1 == label
        truth_label = labels2 == label

        dice = dice_coefficient(pred_label, truth_label)
        dice_coefficients.append(dice)

        #print(f"{label}\t{dice:.4f}")

    return dice_coefficients, all_labels



def find_overlapping_and_neighboring_labels(lesion_mask_path, labels_file_path):
    # Load lesion mask and labels NIfTI files
    lesion_mask = nib.load(lesion_mask_path)
    labels_img = nib.load(labels_file_path)

    # Resample lesion mask to match voxel resolution of labels
    #lesion_mask_resampled = resample_to_img(lesion_mask, labels_img)

    #plot_anat(lesion_mask_resampled, title="Resampled lesion mask",cut_coords=(96, 88,102))
    #plot_anat(labels_img, title="Labels",cut_coords=(96, 88, 102))
    #plot_anat(lesion_mask, title="Original lesion mask",cut_coords=(96, 88, 102))

    #plt.show()


    # Get data arrays from resampled images
    lesion_data = np.int16(lesion_mask.get_fdata())
    labels_data = np.mod(np.int16(labels_img.get_fdata()), 1000)

    # Find labels that overlap with the lesion
    overlapping_labels = set(np.unique(labels_data[lesion_data > 0]))
    overlapping_labels.difference_update(set([0])) # Remove background label

    # Get neighboring labels for each overlapping label
    neighboring_labels = set()
    for label_id in overlapping_labels:
        label_coords = np.argwhere(labels_data == label_id)

        for coord in label_coords:
            x, y, z = coord
            neighbors = [
                labels_data[x + i, y + j, z + k]
                for i in [-1, 0, 1]
                for j in [-1, 0, 1]
                for k in [-1, 0, 1]
                if 0 <= x + i < labels_data.shape[0]
                and 0 <= y + j < labels_data.shape[1]
                and 0 <= z + k < labels_data.shape[2]
            ]
            neighboring_labels.update(neighbors)

    # Remove the original overlapping labels from the set
    neighboring_labels.difference_update(overlapping_labels)
    neighboring_labels.difference_update(set([0])) # Remove background label

    return list(overlapping_labels), list(neighboring_labels)

overlapping_dice_coefficients_avg=[]
neighboring_dice_coefficients_avg=[]
overlapping_dice_coefficients_avg2=[]
neighboring_dice_coefficients_avg2=[]

for subno in range(0,11):
    if subno == 2:
        continue

    # Replace 'label_file1.nii' and 'label_file2.nii' with your actual file paths
    label_file1 = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_orig_BrainSuite/Subject_{subno}_orig.svreg.label.nii.gz"
    label_file3 = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_moved_BrainSuite/Subject_{subno}_moved.svreg.label.nii.gz"

    label_file2 = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_orig_wolesion_BrainSuite/Subject_{subno}_orig_wolesion.svreg.label.nii.gz"
    #label_file2 = "/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_0_inpainted_BrainSuite/Subject_0_inpainted.svreg.label.nii.gz"


    lesion_file = f"/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_lesion_mask.nii.gz"


    overlapping_labels, neighboring_labels = find_overlapping_and_neighboring_labels(lesion_file, label_file2)


    #print(f"Overlapping labels: {overlapping_labels}")
    #print(f"Neighboring labels: {neighboring_labels}")


    dice_coefficients, all_labels = calculate_dice_coefficients(label_file1, label_file2)
    #print(f"Average Dice Coefficient: {np.mean(dice_coefficients):.4f}")


    all_labels = list(all_labels)
    # find indices of overlapping labels in all_labels
    overlapping_label_indices = [all_labels.index(label_id) for label_id in overlapping_labels]

    # find dice coefficients for overlapping labels from these indices
    overlapping_dice_coefficients = [dice_coefficients[i] for i in overlapping_label_indices]

    # find indices of neighboring labels in all_labels
    neighboring_label_indices = [all_labels.index(label_id) for label_id in neighboring_labels]

    # find dice coefficients for neighboring labels from these indices
    neighboring_dice_coefficients = [dice_coefficients[i] for i in neighboring_label_indices]

    # find average dice coefficients for overlapping and neighboring labels
    overlapping_dice_coefficients_avg.append(np.mean(overlapping_dice_coefficients))
    neighboring_dice_coefficients_avg.append(np.mean(neighboring_dice_coefficients))

    #print(f"Average Dice Coefficient for overlapping labels: {overlapping_dice_coefficients_avg:.4f}")
    #print(f"Average Dice Coefficient for neighboring labels: {neighboring_dice_coefficients_avg:.4f}")

    dice_coefficients, all_labels = calculate_dice_coefficients(label_file3, label_file2)
    #print(f"Average Dice Coefficient: {np.mean(dice_coefficients):.4f}")


    all_labels = list(all_labels)
    # find indices of overlapping labels in all_labels
    overlapping_label_indices = [all_labels.index(label_id) for label_id in overlapping_labels]


    # find dice coefficients for overlapping labels from these indices
    overlapping_dice_coefficients = [dice_coefficients[i] for i in overlapping_label_indices]

    # find indices of neighboring labels in all_labels
    neighboring_label_indices = [all_labels.index(label_id) for label_id in neighboring_labels]

    # find dice coefficients for neighboring labels from these indices
    neighboring_dice_coefficients = [dice_coefficients[i] for i in neighboring_label_indices]

    # find average dice coefficients for overlapping and neighboring labels
    overlapping_dice_coefficients_avg2.append(np.mean(overlapping_dice_coefficients))
    neighboring_dice_coefficients_avg2.append(np.mean(neighboring_dice_coefficients))

    #print(f"Average Dice Coefficient for overlapping labels: {overlapping_dice_coefficients_avg2:.4f}")
    #print(f"Average Dice Coefficient for neighboring labels: {neighboring_dice_coefficients_avg2:.4f}")


    print('**********************************************************************************************')


print(f"Average Dice Coefficient for overlapping labels: {np.mean(overlapping_dice_coefficients_avg):.4f}")
print(f"Average Dice Coefficient for neighboring labels: {np.mean(neighboring_dice_coefficients_avg):.4f}")
print(f"Average Dice Coefficient for overlapping labels: {np.mean(overlapping_dice_coefficients_avg2):.4f}")
print(f"Average Dice Coefficient for neighboring labels: {np.mean(neighboring_dice_coefficients_avg2):.4f}")

print('**********************************************************************************************')

# print a table of dice coefficients for each subject
print("Subject\tOverlapping\tNeighboring \t Overlapping\tNeighboring")
print("-------\t-----------\t-----------")
for i in range(len(overlapping_dice_coefficients_avg)):
    print(f"{i}\t{overlapping_dice_coefficients_avg[i]:.4f}\t\t{neighboring_dice_coefficients_avg[i]:.4f}\t {overlapping_dice_coefficients_avg2[i]:.4f}\t\t{neighboring_dice_coefficients_avg2[i]:.4f}")

print('**********************************************************************************************')

 




