#AUM
#Shree Ganeshaya Namaha

import SimpleITK as sitk
import numpy as np

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
    labels1 = read_nifti(label_file1)
    labels2 = read_nifti(label_file2)

    num_labels = max(np.max(labels1), np.max(labels2)) + 1
    dice_coefficients = []

    print("Label\tDice Coefficient")
    for label in range(1, num_labels):
        pred_label = (labels1 == label)
        truth_label = (labels2 == label)
        
        dice = dice_coefficient(pred_label, truth_label)
        dice_coefficients.append(dice)
        
        print(f"{label}\t{dice:.4f}")

    return dice_coefficients

# Replace 'label_file1.nii' and 'label_file2.nii' with your actual file paths
label_file1 = '/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_0_orig_BrainSuite/Subject_0_orig.svreg.label.nii.gz'
label_file2 = '/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_0_inpainted_BrainSuite/Subject_0_inpainted.svreg.label.nii.gz'

dice_coefficients = calculate_dice_coefficients(label_file1, label_file2)
print(f"Average Dice Coefficient: {np.mean(dice_coefficients):.4f}")

