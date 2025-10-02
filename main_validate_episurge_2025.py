# %% [markdown]
# # plot dice Coeff
# 

# %%
import csv
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = '/home/ajoshi/project2_ajoshi_27/data/EPISURG/subjects/'


# %%

def get_dice(seg1_path, seg2_path):
    # Read Nifti files
    seg1 = sitk.ReadImage(seg1_path)
    seg2 = sitk.ReadImage(seg2_path)

    # Convert images to NumPy arrays
    seg1_array = sitk.GetArrayFromImage(seg1)
    seg2_array = sitk.GetArrayFromImage(seg2)

    # Flatten arrays for Dice coefficient calculation
    seg1_flat = seg1_array.flatten() > 0
    seg2_flat = seg2_array.flatten() > 0

    # Calculate Dice coefficient
    dice_coefficient = 2 * np.sum(seg1_flat & seg2_flat) / (np.sum(seg1_flat) + np.sum(seg2_flat))

    print('Dice coefficient: ', dice_coefficient)
    return dice_coefficient



# %%

BASE_PATH = os.environ.get(
    'EPISURG_BASE_PATH',
    '/home/ajoshi/project2_ajoshi_27'
)

CSV_FILE = os.path.join(BASE_PATH, 'data/EPISURG/subjects.csv')
SUBJECTS_DIR = os.path.join(BASE_PATH, 'data/EPISURG/subjects')

# Initialize lists to store Dice scores
dice_manual1 = []
dice_manual2 = []
dice_manual3 = []

# Initialize list to store subject IDs with preop MRI
subjects_with_mri = []

print(f"Reading CSV file: {CSV_FILE}")
print(f"Base subjects directory: {SUBJECTS_DIR}")

# Open the CSV file for reading
try:
    with open(CSV_FILE, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip header row if present
        print(f"CSV headers: {header}")
        
        for row_idx, row in enumerate(csv_reader):
            if len(row) < 7:  # Ensure row has enough columns
                print(f"Skipping row {row_idx} - insufficient columns: {row}")
                continue
                
            subject_id = row[0]
            print(f"Processing subject: {subject_id}")
            
            # Build file paths
            subject_dir = os.path.join(SUBJECTS_DIR, subject_id)
            preop_dir = os.path.join(subject_dir, 'preop')
            postop_dir = os.path.join(subject_dir, 'postop')
            
            preop_mri = os.path.join(preop_dir, f'{subject_id}_preop-t1mri-1.nii.gz')
            postop_mri = os.path.join(postop_dir, f'{subject_id}_postop-t1mri-1.nii.gz')
            
            resection_preop_file = os.path.join(preop_dir, f'{subject_id}_preop-t1mri-1.resection.mask.nii.gz')
            resection_postop_file = os.path.join(postop_dir, f'{subject_id}_postop-t1mri-1.resection.mask.nii.gz')
            
            # Check if preop MRI exists
            if os.path.exists(preop_mri):
                subjects_with_mri.append(subject_id)
            
            # Check if resection mask exists before processing
            if not os.path.exists(resection_postop_file):
                print(f"  Skipping {subject_id} - no postop resection mask found")
                continue
            
            # Process manual segmentation 1
            if len(row) > 4 and row[4] == 'True':
                manual1_resection_file = os.path.join(postop_dir, f'{subject_id}_postop-seg-1.nii.gz')
                if os.path.exists(manual1_resection_file):
                    try:
                        d = get_dice(manual1_resection_file, resection_postop_file)
                        if d > 0.1:
                            dice_manual1.append(d)
                            print(f"  Manual seg 1 Dice: {d:.3f}")
                    except Exception as e:
                        print(f"  Error processing manual seg 1 for {subject_id}: {e}")
            
            # Process manual segmentation 2
            if len(row) > 5 and row[5] == 'True':
                manual2_resection_file = os.path.join(postop_dir, f'{subject_id}_postop-seg-2.nii.gz')
                if os.path.exists(manual2_resection_file):
                    try:
                        d = get_dice(manual2_resection_file, resection_postop_file)
                        if d > 0.1:
                            dice_manual2.append(d)
                            print(f"  Manual seg 2 Dice: {d:.3f}")
                    except Exception as e:
                        print(f"  Error processing manual seg 2 for {subject_id}: {e}")
            
            # Process manual segmentation 3
            if len(row) > 6 and row[6] == 'True':
                manual3_resection_file = os.path.join(postop_dir, f'{subject_id}_postop-seg-3.nii.gz')
                if os.path.exists(manual3_resection_file):
                    try:
                        d = get_dice(manual3_resection_file, resection_postop_file)
                        if d > 0.15:
                            dice_manual3.append(d)
                            print(f"  Manual seg 3 Dice: {d:.3f}")
                    except Exception as e:
                        print(f"  Error processing manual seg 3 for {subject_id}: {e}")

except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILE}")
    print("Please check the path and ensure the file exists.")
except Exception as e:
    print(f"Error reading CSV file: {e}")

# Print summary statistics
print("Subjects with preop MRI available:")
print(subjects_with_mri)
print(f"Number of subjects with preop MRI available: {len(subjects_with_mri)}")

# %% [markdown]
# # plot the dice

# %%
print("Dice score statistics:")
print(f"Manual 1 - Count: {len(dice_manual1)}, Mean: {np.mean(dice_manual1):.3f}" if dice_manual1 else "Manual 1 - No data")
print(f"Manual 2 - Count: {len(dice_manual2)}, Mean: {np.mean(dice_manual2):.3f}" if dice_manual2 else "Manual 2 - No data")
print(f"Manual 3 - Count: {len(dice_manual3)}, Mean: {np.mean(dice_manual3):.3f}" if dice_manual3 else "Manual 3 - No data")

# %%
if dice_manual1 or dice_manual2 or dice_manual3:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    plt.tight_layout(pad=0.4, w_pad=2.5, h_pad=1.0)

    # Plot histogram for manual 1
    if dice_manual1:
        axes[0].hist(dice_manual1, bins=10, range=(0, 1), edgecolor='black')
        axes[0].set_title(f'Researcher in neuroimaging (n={len(dice_manual1)})')
        axes[0].set_xlabel('Dice Coefficient')
        axes[0].set_ylabel('Frequency')
        max_freq_1 = max(np.histogram(dice_manual1, bins=10, range=(0, 1))[0])
        axes[0].set_yticks(np.arange(0, max_freq_1 + 2))
    else:
        axes[0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Researcher in neuroimaging (n=0)')

    # Plot histogram for manual 2
    if dice_manual2:
        axes[1].hist(dice_manual2, bins=10, range=(0, 1), edgecolor='black')
        axes[1].set_title(f'Clinical scientist (n={len(dice_manual2)})')
        axes[1].set_xlabel('Dice Coefficient')
        axes[1].set_ylabel('Frequency')
        max_freq_2 = max(np.histogram(dice_manual2, bins=10, range=(0, 1))[0])
        axes[1].set_yticks(np.arange(0, max_freq_2 + 2))
    else:
        axes[1].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Clinical scientist (n=0)')

    # Plot histogram for manual 3
    if dice_manual3:
        axes[2].hist(dice_manual3, bins=10, range=(0, 1), edgecolor='black')
        axes[2].set_title(f'Neurologist (n={len(dice_manual3)})')
        axes[2].set_xlabel('Dice Coefficient')
        axes[2].set_ylabel('Frequency')
        max_freq_3 = max(np.histogram(dice_manual3, bins=10, range=(0, 1))[0])
        axes[2].set_yticks(np.arange(0, max_freq_3 + 2))
    else:
        axes[2].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Neurologist (n=0)')

    plt.tight_layout()
    plt.savefig('dice_manual_full.png', dpi=300, bbox_inches='tight')
    print("Saved plot as 'dice_manual_full.png'")
    plt.show()
else:
    print("No dice scores available to plot.") 


