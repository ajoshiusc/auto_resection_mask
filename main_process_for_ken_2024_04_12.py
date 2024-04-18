
import glob
from autoresec import delineate_resection, delineate_resection_post
import os


sublist = glob.glob('/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_12_mri_dump/*')

# Initialize an empty list to store subject IDs with preop MRI
subjects_with_mri = []

#sublist = ['M1985N39'] #['F2001K75']
# Open the CSV file for reading
for fname in sublist:
    subid = os.path.basename(fname)
    preop_mri = glob.glob(f'/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_12_mri_dump/{subid}/sMRI/*{subid}?MRI.nii*') #f'/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_12_mri_dump/{subid}/sMRI/sub-{subid}-{subid}_MRI.nii.gz'
    postop_mri = glob.glob(f'/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_12_mri_dump/{subid}/sMRI/*{subid}*post*.nii*') #f'/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_12_mri_dump/{subid}/sMRI/sub-{subid}-{subid}_post_RS_MRI.nii.gz'

    if len(postop_mri) == 0 or len(preop_mri) == 0:
        continue
    else:
        preop_mri = preop_mri[0]
        postop_mri = postop_mri[0]

    # if extension is .nii, compress the file and make it .nii.gz
    if preop_mri.endswith('.nii'):
        os.system(f'gzip {preop_mri}')
        preop_mri = preop_mri + '.gz'

    if postop_mri.endswith('.nii'):
        os.system(f'gzip {postop_mri}')
        postop_mri = postop_mri + '.gz'


    # Check if both preop and postop MRI exist
    if os.path.isfile(preop_mri) and os.path.isfile(postop_mri):


        #if not os.path.isfile(postop_mri.replace('.nii.gz', '.resection.mask.nii.gz')):
        #    delineate_resection_post(preop_mri, postop_mri)
        
        if not os.path.isfile(preop_mri.replace('.nii.gz', '.resection.mask.nii.gz')):

            try:
                delineate_resection(preop_mri, postop_mri)
            except Exception as e:
                print(f'Error processing subject {subid}: {e}')
                continue
        else:
            print(f'Subject {subid} ALREADY processed')

        #delineate_resection(preop_mri, postop_mri)

        print(f'Subject {subid} processing done!')



print('Done')



    

