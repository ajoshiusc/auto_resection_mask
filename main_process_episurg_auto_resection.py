import csv
import os
import shutil
from auto_resection_mask import auto_resection_mask

# Specify the file path
csv_file = '/deneb_disk/auto_resection/EPISURG/subjects.csv'
brainsuite_path = "/home/ajoshi/Software/BrainSuite23a"
bst_atlas_path = "bst_atlases/icbm_bst.nii.gz"
bst_atlas_labels_path = "bst_atlases/icbm_bst.label.nii.gz"
surrogate_preop = "/home/ajoshi/Software/BrainSuite23a/svreg/BrainSuiteAtlas1/mri.nii.gz"  # Surrogate pre-op scan

# Initialize an empty list to store processed subjects
processed_subjects = []
failed_subjects = []

print("Starting EPISURG auto resection mask processing...")
print(f"Using BrainSuite path: {brainsuite_path}")
print(f"Using surrogate preop scan: {surrogate_preop}")
print("-" * 60)

# Open the CSV file for reading
with open(csv_file, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)

    # Iterate through the rows in the CSV file
    for row in csv_reader:
        subject_id = row[0]
        has_preop = row[3] == 'True'
        has_postop = row[4] == 'True' or row[5] == 'True' or row[6] == 'True'
        
        # Only process if postop is available
        if has_postop:
            print(f"Processing subject {subject_id}...")
            
            # Determine preop MRI path
            if has_preop:
                preop_mri = f'/deneb_disk/auto_resection/EPISURG/subjects/{subject_id}/preop/{subject_id}_preop-t1mri-1.nii.gz'
                if os.path.isfile(preop_mri):
                    print(f"  Using actual preop scan for {subject_id}")
                else:
                    print(f"  Preop scan not found for {subject_id}, using surrogate")
                    # Copy surrogate to subject directory
                    subject_preop_dir = f'/deneb_disk/auto_resection/EPISURG/subjects/{subject_id}/preop'
                    os.makedirs(subject_preop_dir, exist_ok=True)
                    surrogate_copy = f'{subject_preop_dir}/{subject_id}_surrogate_preop.nii.gz'
                    shutil.copy2(surrogate_preop, surrogate_copy)
                    preop_mri = surrogate_copy
                    print(f"  Copied surrogate preop to {surrogate_copy}")
            else:
                # Copy surrogate to subject directory
                subject_preop_dir = f'/deneb_disk/auto_resection/EPISURG/subjects/{subject_id}/preop'
                os.makedirs(subject_preop_dir, exist_ok=True)
                surrogate_copy = f'{subject_preop_dir}/{subject_id}_surrogate_preop.nii.gz'
                
                # Only copy if it doesn't already exist
                if not os.path.isfile(surrogate_copy):
                    shutil.copy2(surrogate_preop, surrogate_copy)
                    print(f"  Copied surrogate preop to {surrogate_copy}")
                else:
                    print(f"  Using existing surrogate preop at {surrogate_copy}")
                preop_mri = surrogate_copy
            
            # Postop MRI path
            postop_mri = f'/deneb_disk/auto_resection/EPISURG/subjects/{subject_id}/postop/{subject_id}_postop-t1mri-1.nii.gz'
            
            # Check if postop file exists
            if not os.path.isfile(postop_mri):
                print(f"  Warning: Postop MRI not found for {subject_id}, skipping...")
                failed_subjects.append((subject_id, "Postop MRI not found"))
                continue
            
            # Check if already processed (look for resection mask files)
            # Generate mask filename based on the actual preop file being used
            preop_base = os.path.splitext(os.path.splitext(os.path.basename(preop_mri))[0])[0]  # Remove .nii.gz
            preop_mask = f'/deneb_disk/auto_resection/EPISURG/subjects/{subject_id}/preop/{preop_base}.resection.mask.nii.gz'
            postop_mask = f'/deneb_disk/auto_resection/EPISURG/subjects/{subject_id}/postop/{subject_id}_postop-t1mri-1.resection.mask.nii.gz'
            
            if os.path.isfile(preop_mask) and os.path.isfile(postop_mask):
                print(f'  Subject {subject_id} already processed, skipping....')
                processed_subjects.append(subject_id)
                continue
            
            try:
                # Run auto_resection_mask
                print(f"  Running auto_resection_mask for {subject_id}...")
                auto_resection_mask(
                    preop_mri, 
                    postop_mri, 
                    BrainSuitePATH=brainsuite_path,
                    bst_atlas_path=bst_atlas_path,
                    bst_atlas_labels_path=bst_atlas_labels_path
                )
                processed_subjects.append(subject_id)
                print(f"  ✓ Successfully processed subject {subject_id}")
                
            except (OSError, RuntimeError, ValueError) as e:
                error_msg = f"Error processing: {str(e)}"
                print(f"  ✗ Error processing subject {subject_id}: {error_msg}")
                failed_subjects.append((subject_id, error_msg))
                continue
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                print(f"  ✗ Unexpected error processing subject {subject_id}: {error_msg}")
                failed_subjects.append((subject_id, error_msg))
                continue
                
        else:
            print(f"Subject {subject_id} has no postop scan available, skipping...")
            failed_subjects.append((subject_id, "No postop scan available"))

# Print summary
print("\n" + "=" * 60)
print("PROCESSING COMPLETE!")
print("=" * 60)
print(f"Successfully processed subjects ({len(processed_subjects)}):")
for subject in processed_subjects:
    print(f"  ✓ {subject}")

if failed_subjects:
    print(f"\nFailed subjects ({len(failed_subjects)}):")
    for subject, reason in failed_subjects:
        print(f"  ✗ {subject}: {reason}")

print(f"\nTotal subjects processed: {len(processed_subjects)}")
print(f"Total subjects failed: {len(failed_subjects)}")
print("Done!")