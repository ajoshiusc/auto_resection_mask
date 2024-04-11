
import csv
import glob

from cv2 import Subdiv2D
from requests import post
from autoresec import delineate_resection, delineate_resection_post
import os


sublist = glob.glob('/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_10_mri_dump/*')

# Initialize an empty list to store subject IDs with preop MRI
subjects_with_mri = []

# Open the CSV file for reading
for fname in sublist:
    subid = os.path.basename(fname)
    preop_mri = f'/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_10_mri_dump/{subid}/sMRI/sub-{subid}-{subid}_MRI.nii.gz'
    postop_mri = f'/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_10_mri_dump/{subid}/sMRI/sub-{subid}-{subid}_post_RS_MRI.nii.gz'

    if not os.path.isfile(postop_mri):
        postop_mri = f'/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_10_mri_dump/{subid}/sMRI/sub-{subid}-{subid}_postRS_MRI.nii.gz'

    # Check if the subject has preop MRI
    if os.path.isfile(preop_mri):
        print(f'Subject {subid} has preop MRI')
    else:
        print(f'File {preop_mri} does not exist')
    
    # Check if the subject has postop MRI
    if os.path.isfile(postop_mri):
        print(f'Subject {subid} has postop MRI')
    else:
        print(f'File {postop_mri} does not exist')           


    # Check if both preop and postop MRI exist
    if os.path.isfile(preop_mri) and os.path.isfile(postop_mri):


        #if not os.path.isfile(postop_mri.replace('.nii.gz', '.resection.mask.nii.gz')):
        #    delineate_resection_post(preop_mri, postop_mri)
        
        #if not os.path.isfile(preop_mri.replace('.nii.gz', '.resection.mask.nii.gz')):

        try:
            delineate_resection(preop_mri, postop_mri)
        except Exception as e:
            print(f'Error processing subject {subid}: {e}')
            continue

        delineate_resection(preop_mri, postop_mri)

        print(f'Subject {subid} processed')



print('Done')



    

