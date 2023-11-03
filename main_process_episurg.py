import csv
from autoresec import delineate_resection, delineate_resection_post
import os
# Specify the file path
csv_file = '/deneb_disk/EPISURG/subjects.csv'  # Replace with your CSV file path

# Initialize an empty list to store subject IDs with preop MRI
subjects_with_mri = []

# Open the CSV file for reading
with open(csv_file, mode='r') as file:
    csv_reader = csv.reader(file)

    # Iterate through the rows in the CSV file
    for row in csv_reader:
        # Check if the 4th column (index 3 in 0-based indexing) contains "1" (indicating preop MRI available)
        if (row[3] == 'True') and (row[4] == 'True' or row[5] == 'True' or row[6] == 'True'):

            print(f'{row[4]} {row[5]} {row[6]}')
            # Add the subject ID (1st column) to the list
            subjects_with_mri.append(row[0])

            preop_mri = '/deneb_disk/EPISURG/subjects/' + \
                row[0] + '/preop/' + row[0] + '_preop-t1mri-1.nii.gz'
            postop_mri = '/deneb_disk/EPISURG/subjects/' + \
                row[0] + '/postop/' + row[0] + '_postop-t1mri-1.nii.gz'

            if not os.path.isfile('/deneb_disk/EPISURG/subjects/' + row[0] + '/preop/' + row[0] + '_preop-t1mri-1.resection.mask.nii.gz'):
                delineate_resection(preop_mri, postop_mri)
            else:
                print(f'Subject pre {row[0]} already processed, skipping ....')


            if not os.path.isfile('/deneb_disk/EPISURG/subjects/' + row[0] + '/postop/' + row[0] + '_postop-t1mri-1.resection.mask.nii.gz'):
                delineate_resection_post(preop_mri, postop_mri)
            else:
                print(f'Subject post {row[0]} already processed, skipping ....')


# Print the list of subjects with preop MRI
print("Subjects with preop MRI available:")
print(subjects_with_mri)
print("Number of subjects with preop MRI available: {}".format(
    len(subjects_with_mri)))
