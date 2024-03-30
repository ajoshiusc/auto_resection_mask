
import csv

#from cv2 import Subdiv2D
#from requests import post
from autoresec import delineate_resection, delineate_resection_post
import os

preop_mri = '/deneb_disk/auto_resection/data_8_4_2023/sub-M1998N33/sMRI/sub-M1998N33-M1998N33_MRI.nii.gz'
postop_mri = '/deneb_disk/auto_resection/data_8_4_2023/sub-M1998N33/sMRI/sub-M1998N33-M1998N33_postRS_MRI.nii.gz'

delineate_resection_post(preop_mri, postop_mri)
delineate_resection(preop_mri, postop_mri)



print('Done')



    

