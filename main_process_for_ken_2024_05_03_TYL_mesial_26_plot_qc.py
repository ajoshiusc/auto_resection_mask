
import glob

import os
from nilearn.plotting import plot_anat, find_xyz_cut_coords
from matplotlib import pyplot as plt
import numpy as np
import nibabel as nb


def process_subject(fname, outdir):
    subid = os.path.basename(fname)
    preop_mri = glob.glob(f'{outdir}/{subid}/*{subid}?MRI.nii*')
    postop_mri = glob.glob(f'{outdir}/{subid}/*{subid}*post*.nii*')

    #if os.path.isfile(f'sub-{subid}_resection.png'):
    #    print(f'File sub-{subid}_resection.png already exists')
    #    return

    if len(postop_mri) == 0 or len(preop_mri) == 0:
        return
    else:
        preop_mri = preop_mri[0]
        postop_mri = postop_mri[0]

    if os.path.isfile(preop_mri) and os.path.isfile(postop_mri):
        if os.path.isfile(preop_mri.replace('.nii.gz', '.resection.mask.nii.gz')):
            resection_mask = preop_mri.replace('.nii.gz', '.resection.mask.nii.gz')
            
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))
            cut_coords = find_xyz_cut_coords(resection_mask)

            img = nb.load(preop_mri).get_fdata()
            pctl = np.percentile(img, 99)
            p = plot_anat(preop_mri, title=f'{subid} preop MRI with resection mask', axes=axes[0], cut_coords=cut_coords, vmax=pctl)
            p.add_contours(resection_mask, levels=[.5], colors='r')

            post2pre_mri = preop_mri.replace('.nii.gz', '.affine.post2pre.nii.gz')
            post2pre_mri_nonlin = preop_mri.replace('.nii.gz', '.nonlin.post2pre.nii.gz')

            img = nb.load(post2pre_mri).get_fdata()
            pctl = np.percentile(img, 99)
            plot_anat(post2pre_mri, title=f'{subid} post2preop MRI (affine)', axes=axes[1], cut_coords=cut_coords, vmax=pctl)

            img = nb.load(post2pre_mri_nonlin).get_fdata()
            pctl = np.percentile(img, 99)
            plot_anat(post2pre_mri_nonlin, title=f'{subid} post2preop MRI (nonlinear)', axes=axes[2], cut_coords=cut_coords, vmax=pctl)

            fig.savefig(f'{outdir}/{subid}/sub-{subid}-{subid}_MRI_resection.png')
            fig.savefig(f'sub-{subid}_resection.png')
            print(f'Plotted preop MRI with resection mask for {subid}')
            plt.close()
        else:
            print(f'No resection mask found for {subid}')
    else:
        print(f'Either preop or postop MRI is missing for {subid}')

# Specify the output directory

outdir = '/deneb_disk/auto_resection/seizure_free_patients_from_ken/TLY_mesial_26_sub_6_01_2024'
os.makedirs(outdir, exist_ok=True)
#sublist = glob.glob('/deneb_disk/auto_resection/seizure_free_patients_from_ken/2024_04_12_mri_dump/*')

# read the list of subjects from a txt file
sublist = open('TL_mesial_26.txt', 'r', encoding='utf-8').read().splitlines()

# Initialize an empty list to store subject IDs with preop MRI
subjects_with_mri = []

# Open the CSV file for reading
for fname in sublist:
    process_subject(fname, outdir)



print('Done')



    

