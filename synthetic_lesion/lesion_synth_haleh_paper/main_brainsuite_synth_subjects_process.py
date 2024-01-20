import nibabel as nib
import numpy as np
from multiprocessing import Pool
import shutil
import os


def process_img(img_file):


    dir_name = img_file.replace('.nii.gz','_BrainSuite')
    print(dir_name)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    shutil.copy(img_file,dir_name)

    _, fname = os.path.split(img_file)

    img_file = os.path.join(dir_name,fname)

    stats_file = img_file.replace('.nii.gz','.roiwise.stats.txt')

    if os.path.exists(stats_file):
        return

    out_msk_file = img_file.replace('.nii.gz','.mask.nii.gz')
    bse_file = img_file.replace('.nii.gz','.bse.nii.gz')

    img = nib.load(img_file)

    mask = nib.Nifti1Image(255*(img.get_fdata() > 2).astype(np.uint8),img.affine)
    nib.save(mask, out_msk_file)

    bse_img = nib.Nifti1Image(((img.get_fdata() > 2)*img.get_fdata()).astype(np.int16),img.affine)
    nib.save(bse_img, bse_file)

    cmd = '/home/ajoshi/Software/BrainSuite23a/bin/brainsuite_anatomical_pipeline_nobse.sh ' + img_file
    print(cmd)
    os.system(cmd)



file_list = []

for subno in range(27,32):

    file_list.append(f'/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_orig.nii.gz')
    file_list.append(f'/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_inpainted.nii.gz')
    #file_list.append(f'/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_moved.nii.gz')
    file_list.append(f'/deneb_disk/Inpainting_Lesions_Examples/brainsuite_synth_lesion/Subject_{subno}_orig_wolesion.nii.gz')


pool = Pool(processes=3)

print('++++++++++++++')
#pool.map(process_img, file_list)
print('++++SUBMITTED++++++')

for f in file_list:
    process_img(f)

pool.close()
pool.join()

