import csv
import os
import shutil
from auto_resection_mask import auto_resection_mask

# Configuration paths
episurg_dir = '/deneb_disk/auto_resection/EPISURG'
csv_file = os.path.join(episurg_dir, 'subjects.csv')
brainsuite_path = "/home/ajoshi/Software/BrainSuite23a"
bst_atlas_path = "bst_atlases/icbm_bst.nii.gz"
bst_atlas_labels_path = "bst_atlases/icbm_bst.label.nii.gz"
surrogate_preop = os.path.join(brainsuite_path, "svreg/BrainSuiteAtlas1/mri.nii.gz")  # Surrogate pre-op scan

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
        # Skip header row if it exists
        if row[0] == 'Subject':
            continue
            
        subject_id = row[0]
        # More robust boolean parsing for preop availability
        has_preop = str(row[3]).strip().lower() in ['true', '1', 'yes']
        
        # Check if postop scan actually exists on disk (not relying on CSV rater columns)
        subject_dir = os.path.join(episurg_dir, 'subjects', subject_id)
        subject_postop_dir = os.path.join(subject_dir, 'postop')
        postop_mri = os.path.join(subject_postop_dir, f'{subject_id}_postop-t1mri-1.nii.gz')
        has_postop = os.path.isfile(postop_mri)
        
        print(f"Subject {subject_id}: preop_csv={row[3]}, has_preop={has_preop}, has_postop={has_postop}")
        
        # Only process if postop is available
        if has_postop:
            print(f"Processing subject {subject_id}...")
            
            subject_preop_dir = os.path.join(subject_dir, 'preop')
            
            if has_preop:
                preop_mri = os.path.join(subject_preop_dir, f'{subject_id}_preop-t1mri-1.nii.gz')
                if os.path.isfile(preop_mri):
                    print(f"  Using actual preop scan for {subject_id}")
                else:
                    print(f"  Preop scan not found for {subject_id}, using surrogate")
                    # Copy surrogate to subject directory
                    os.makedirs(subject_preop_dir, exist_ok=True)
                    surrogate_copy = os.path.join(subject_preop_dir, f'{subject_id}_surrogate_preop.nii.gz')
                    shutil.copy2(surrogate_preop, surrogate_copy)
                    preop_mri = surrogate_copy
                    print(f"  Copied surrogate preop to {surrogate_copy}")
            else:
                # Copy surrogate to subject directory
                os.makedirs(subject_preop_dir, exist_ok=True)
                surrogate_copy = os.path.join(subject_preop_dir, f'{subject_id}_surrogate_preop.nii.gz')
                
                # Only copy if it doesn't already exist
                if not os.path.isfile(surrogate_copy):
                    shutil.copy2(surrogate_preop, surrogate_copy)
                    print(f"  Copied surrogate preop to {surrogate_copy}")
                else:
                    print(f"  Using existing surrogate preop at {surrogate_copy}")
                preop_mri = surrogate_copy
            
            # Check if already processed (look for resection mask files)
            # Generate mask filename based on the actual preop file being used
            preop_base = os.path.splitext(os.path.splitext(os.path.basename(preop_mri))[0])[0]  # Remove .nii.gz
            preop_mask = os.path.join(subject_preop_dir, f'{preop_base}.resection.mask.nii.gz')
            postop_mask = os.path.join(subject_postop_dir, f'{subject_id}_postop-t1mri-1.resection.mask.nii.gz')
            
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