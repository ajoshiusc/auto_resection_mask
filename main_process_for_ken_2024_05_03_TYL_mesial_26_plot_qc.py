
import glob

import os
from nilearn.plotting import plot_anat, find_xyz_cut_coords
from matplotlib import pyplot as plt
import numpy as np
import nibabel as nb


outdir = '/deneb_disk/auto_resection/seizure_free_patients_from_ken/TLY_mesial_26_sub_5_25_2024'
os.makedirs(outdir, exist_ok=True)
#sublist = glob.glob('/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_12_mri_dump/*')

# read the list of subjects from a txt file
sublist = open('TL_mesial_26.txt', 'r', encoding='utf-8').read().splitlines()

# Initialize an empty list to store subject IDs with preop MRI
subjects_with_mri = []

# Open the CSV file for reading
for fname in sublist:
    subid = os.path.basename(fname)
    preop_mri = glob.glob(f'{outdir}/{subid}/*{subid}?MRI.nii*') #f'/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_12_mri_dump/{subid}/sMRI/sub-{subid}-{subid}_MRI.nii.gz'
    postop_mri = glob.glob(f'{outdir}/{subid}/*{subid}*post*.nii*') #f'/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_12_mri_dump/{subid}/sMRI/sub-{subid}-{subid}_post_RS_MRI.nii.gz'

    if os.path.isfile(f'sub-{subid}_resection.png'):
        print(f'File sub-{subid}_resection.png already exists')
        continue

    if len(postop_mri) == 0 or len(preop_mri) == 0:
        continue
    else:
        preop_mri = preop_mri[0]
        postop_mri = postop_mri[0]

    # Check if both preop and postop MRI exist
    if os.path.isfile(preop_mri) and os.path.isfile(postop_mri):


        #if not os.path.isfile(postop_mri.replace('.nii.gz', '.resection.mask.nii.gz')):
        #    delineate_resection_post(preop_mri, postop_mri)
        
        if os.path.isfile(preop_mri.replace('.nii.gz', '.resection.mask.nii.gz')):

            resection_mask = preop_mri.replace('.nii.gz', '.resection.mask.nii.gz')
            
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

            #plot the preop mri with resection mask
            cut_coords = find_xyz_cut_coords(resection_mask)

            # find 90th percentile of the preop MRI
            img = nb.load(preop_mri).get_fdata()
            pctl = np.percentile(img, 99)


            p = plot_anat(preop_mri, title=f'{subid} preop MRI with resection mask', axes=axes[0], cut_coords=cut_coords, vmax=pctl)
            p.add_contours(resection_mask, levels=[.5], colors='r')
            # also plot postop MRI in the same figure

            post2pre_mri = preop_mri.replace('.nii.gz', '.affine.post2pre.nii.gz')
            # find 90th percentile of the postop MRI
            img = nb.load(post2pre_mri).get_fdata()
            pctl = np.percentile(img, 99)
            plot_anat(post2pre_mri, title=f'{subid} post2preop MRI', axes=axes[1], cut_coords=cut_coords, vmax=pctl)
            plt.tight_layout()

            fig.savefig(f'{outdir}/{subid}/sub-{subid}-{subid}_MRI_resection.png')
            fig.savefig(f'sub-{subid}_resection.png')
            #plt.show()
            print(f'Plotted preop MRI with resection mask for {subid}')
            plt.close()
        else:
            print(f'No resection mask found for {subid}')

    else:
        print(f'Either preop or postop MRI is missing for {subid}')



print('Done')



    

