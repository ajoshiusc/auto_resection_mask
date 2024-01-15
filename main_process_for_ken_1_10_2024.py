
import csv

from cv2 import Subdiv2D
from requests import post
from autoresec import delineate_resection, delineate_resection_post
import os

preop_mri = '/home/ajoshi/Desktop/for_ken_1_10_2024/sub-M1998N33/sMRI/sub-M1998N33-M1998N33_MRI_1mm.nii.gz'
postop_mri = '/home/ajoshi/Desktop/for_ken_1_10_2024/sub-M1998N33/sMRI/sub-M1998N33-M1998N33_postRS_MRI_1mm.nii.gz'

delineate_resection_post(preop_mri, postop_mri)
delineate_resection(preop_mri, postop_mri)



print('Done')



    

