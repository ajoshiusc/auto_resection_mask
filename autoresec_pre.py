
import csv

#from cv2 import Subdiv2D
#from requests import post
from autoresec import delineate_resection
import sys

# make delineate_resection a script that can be called from the command line that takes in the preop and postop MRI paths
# and returns the resection mask path

if __name__ == "__main__":
    # the two command line arguments are the preop and postop MRI paths. Print usage if no argument is passed
    if len(sys.argv) != 3:
        print("Usage: python autoresec_pre.py <preop_mri_path> <postop_mri_path>")
        sys.exit(1)

    pre_mri = sys.argv[1]
    post_mri = sys.argv[2]
    delineate_resection(pre_mri, post_mri)







