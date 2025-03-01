# Description: This script is used to generate the resection mask for the preoperative MRI using the postoperative MRI.
from autoresec import delineate_resection_pre, delineate_resection_post
from resection_overlay_plots import generate_resection_overlay_plots



def auto_resection_mask(preop_mri, postop_mri, BrainSuitePATH):
    delineate_resection_post(preop_mri, postop_mri, BrainSuitePATH=BrainSuitePATH)
    delineate_resection_pre(preop_mri, postop_mri, BrainSuitePATH=BrainSuitePATH)
    generate_resection_overlay_plots(preop_mri, postop_mri)

if __name__ == "__main__":
    mypreop_mri = '/home/ajoshi/Downloads/example_dataset_resection/preop.nii.gz'
    mypostop_mri = '/home/ajoshi/Downloads/example_dataset_resection/postop.nii.gz'
    my_brainsuite = "/home/ajoshi/Software/BrainSuite23a"

    auto_resection_mask(mypreop_mri, mypostop_mri, BrainSuitePATH=my_brainsuite)
    print('Done')
