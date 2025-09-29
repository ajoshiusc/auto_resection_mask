import csv
import os
import shutil
import subprocess
from pathlib import Path

# Configuration paths
episurg_dir = '/project2/ajoshi_27/data/EPISURG'
csv_file = os.path.join(episurg_dir, 'subjects.csv')
brainsuite_path = "/project2/ajoshi_27/BrainSuite23a"
bst_atlas_path = "bst_atlases/icbm_bst.nii.gz"
bst_atlas_labels_path = "bst_atlases/icbm_bst.label.nii.gz"
surrogate_preop = os.path.join(brainsuite_path, "svreg/BrainSuiteAtlas1/mri.nii.gz")  # Surrogate pre-op scan

# SBATCH configuration
project_dir = "/project2/ajoshi_27/GitHub/auto_resection_mask"
python3gpu_job_script = os.path.join(project_dir, "python3gpu.job")
main_script = os.path.join(project_dir, "auto_resection_mask.py")
job_output_dir = os.path.join(episurg_dir, 'sbatch_logs')

# Create job output directory if it doesn't exist
os.makedirs(job_output_dir, exist_ok=True)

# Initialize lists to store job submission status
submitted_jobs = []
failed_submissions = []
already_processed = []
no_postop_subjects = []

print("Starting EPISURG auto resection mask processing with SBATCH job submission...")
print(f"Using BrainSuite path: {brainsuite_path}")
print(f"Using surrogate preop scan: {surrogate_preop}")
print(f"Using python3gpu job script: {python3gpu_job_script}")
print(f"Using main script: {main_script}")
print(f"Job logs will be stored in: {job_output_dir}")
print("-" * 60)

def submit_sbatch_job(subject_id, preop_mri, postop_mri):
    """Submit SBATCH job for a single subject using CLI script"""
    job_name = f"auto_resec_{subject_id}"
    log_file = os.path.join(job_output_dir, f"{subject_id}.%j.log")
    
    # Create the command that python3gpu.job will execute
    cli_command = f"{main_script} '{preop_mri}' '{postop_mri}' --brainsuite-path '{brainsuite_path}' --bst-atlas-path '{bst_atlas_path}' --bst-atlas-labels-path '{bst_atlas_labels_path}' --subject-id '{subject_id}'"
    
    # Create a simple wrapper script that the job will execute
    wrapper_script = os.path.join(job_output_dir, f"run_{subject_id}.sh")
    with open(wrapper_script, 'w') as f:
        f.write(f"#!/bin/bash\n{cli_command}\n")
    os.chmod(wrapper_script, 0o755)
    
    # SBATCH command
    sbatch_cmd = [
        "sbatch",
        "--job-name", job_name,
        "--output", log_file,
        "--error", log_file,
        python3gpu_job_script,
        wrapper_script
    ]
    
    try:
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]  # Extract job ID
        print(f"  ✓ Submitted job {job_id} for subject {subject_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to submit job for subject {subject_id}: {e.stderr}")
        return None

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
            print(f"Preparing job for subject {subject_id}...")
            
            subject_preop_dir = os.path.join(subject_dir, 'preop')
            
            # Determine preop MRI path
            if has_preop:
                preop_mri = os.path.join(subject_preop_dir, f'{subject_id}_preop-t1mri-1.nii.gz')
                if os.path.isfile(preop_mri):
                    print(f"  Using actual preop scan for {subject_id}")
                else:
                    print(f"  Preop scan not found for {subject_id}, using surrogate")
                    # Copy surrogate to subject directory
                    os.makedirs(subject_preop_dir, exist_ok=True)
                    surrogate_copy = os.path.join(subject_preop_dir, f'{subject_id}_surrogate_preop.nii.gz')
                    if not os.path.isfile(surrogate_copy):
                        shutil.copy2(surrogate_preop, surrogate_copy)
                        print(f"  Copied surrogate preop to {surrogate_copy}")
                    preop_mri = surrogate_copy
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
                already_processed.append(subject_id)
                continue
            
            # Submit SBATCH job directly using CLI
            job_id = submit_sbatch_job(subject_id, preop_mri, postop_mri)
            
            if job_id:
                submitted_jobs.append((subject_id, job_id))
            else:
                failed_submissions.append((subject_id, "SBATCH submission failed"))
                
        else:
            print(f"Subject {subject_id} has no postop scan available, skipping...")
            no_postop_subjects.append(subject_id)

# Print summary
print("\n" + "=" * 60)
print("JOB SUBMISSION COMPLETE!")
print("=" * 60)
print(f"Successfully submitted jobs ({len(submitted_jobs)}):")
for subject, job_id in submitted_jobs:
    print(f"  ✓ {subject} -> Job ID: {job_id}")

print(f"\nAlready processed subjects ({len(already_processed)}):")
for subject in already_processed:
    print(f"  - {subject}")

if no_postop_subjects:
    print(f"\nSubjects with no postop scan ({len(no_postop_subjects)}):")
    for subject in no_postop_subjects:
        print(f"  - {subject}")

if failed_submissions:
    print(f"\nFailed job submissions ({len(failed_submissions)}):")
    for subject, reason in failed_submissions:
        print(f"  ✗ {subject}: {reason}")

print(f"\nTotal jobs submitted: {len(submitted_jobs)}")
print(f"Total already processed: {len(already_processed)}")
print(f"Total failed submissions: {len(failed_submissions)}")
print(f"Total no postop: {len(no_postop_subjects)}")

print(f"\nJob logs will be available in: {job_output_dir}")
print("Use 'squeue -u $USER' to monitor job status")
print("Done!")