import sys
sys.path.append('/home/ajoshi/Projects/auto_resection_mask')

from auto_resection_mask import auto_resection_mask

preop_mri = '/home/ajoshi/Downloads/example_dataset_resection/preop.nii.gz'
postop_mri = '/home/ajoshi/Downloads/example_dataset_resection/postop.nii.gz'

auto_resection_mask(preop_mri, postop_mri)

print('Done')



    

