
import csv

from cv2 import Subdiv2D
from requests import post
from autoresec import delineate_resection, delineate_resection_post
import os
# Specify the file path
csv_file = '/deneb_disk/auto_resection/data_8_4_2023/test_subjects.csv'  # Replace with your CSV file path

# Initialize an empty list to store subject IDs with preop MRI
subjects_with_mri = []

# Open the CSV file for reading
with open(csv_file, mode='r') as file:
    csv_reader = csv.reader(file)

    # Iterate through the rows in the CSV file
    for row in csv_reader:
        
        subid = row[0]
        preop_mri = f'/deneb_disk/auto_resection/data_8_4_2023/sub-{subid}/sMRI/sub-{subid}-{subid}_MRI.nii.gz'
        postop_mri = f'/deneb_disk/auto_resection/data_8_4_2023/sub-{subid}/sMRI/sub-{subid}-{subid}_post_RS_MRI.nii.gz'

        if not os.path.isfile(postop_mri):
            postop_mri = f'/deneb_disk/auto_resection/data_8_4_2023/sub-{subid}/sMRI/sub-{subid}-{subid}_postRS_MRI.nii.gz'

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

            delineate_resection_post(preop_mri, postop_mri)
            delineate_resection(preop_mri, postop_mri)
            print(f'Subject {subid} processed')



print('Done')



    

