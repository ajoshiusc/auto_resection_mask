
import csv

#from cv2 import Subdiv2D
#from requests import post
from autoresec import delineate_resection, delineate_resection_post
import os

preop_mri = '/home/ajoshi/Downloads/NE_AD/test_run/Pre_T13D_bst.nii.gz'
postop_mri = '/home/ajoshi/Downloads/NE_AD/test_run/Post_T13D_bst.nii.gz'

delineate_resection_post(preop_mri, postop_mri)
delineate_resection(preop_mri, postop_mri)



print('Done')



    

