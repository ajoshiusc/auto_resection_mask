import glob
import os
from resection_overlay_plots import generate_resection_overlay_plots

# Specify the output directory
outdir = '/deneb_disk/auto_resection/seizure_free_patients_from_ken/TLY_mesial_26_sub_6_01_2024'
os.makedirs(outdir, exist_ok=True)

# read the list of subjects from a txt file
sublist = open('TL_mesial_26.txt', 'r', encoding='utf-8').read().splitlines()

# Initialize an empty list to store subject IDs with preop MRI
subjects_with_mri = []

# Open the CSV file for reading
for subid in sublist:
    #postop_mri = fname.replace('preop', 'postop')  # Assuming the naming convention for postop MRI

    if os.path.isdir(f'{outdir}/{subid}') == False:
        continue

    preop_mri = glob.glob(f'{outdir}/{subid}/*{subid}?MRI.nii*')[0] #f'/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_12_mri_dump/{subid}/sMRI/sub-{subid}-{subid}_MRI.nii.gz'
    postop_mri = glob.glob(f'{outdir}/{subid}/*{subid}*post*.nii*')[0]#f'/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_12_mri_dump/{subid}/sMRI/sub-{subid}-{subid}_post_RS_MRI.nii.gz'

    generate_resection_overlay_plots(preop_mri, postop_mri)

print('Done')