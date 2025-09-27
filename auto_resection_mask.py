# Description: This script is used to generate the resection mask for the preoperative MRI using the postoperative MRI.
from autoresec import delineate_resection_pre, delineate_resection_post
from resection_overlay_plots import generate_resection_overlay_plots



def auto_resection_mask(preop_mri, postop_mri, BrainSuitePATH,
                        bst_atlas_path="bst_atlases/icbm_bst.nii.gz",
                        bst_atlas_labels_path="bst_atlases/icbm_bst.label.nii.gz" ):
    
    delineate_resection_post(preop_mri, postop_mri, BrainSuitePATH=BrainSuitePATH,
                            bst_atlas_path=bst_atlas_path,bst_atlas_labels_path=bst_atlas_labels_path)
    delineate_resection_pre(preop_mri, postop_mri, BrainSuitePATH=BrainSuitePATH,
                           bst_atlas_path=bst_atlas_path,bst_atlas_labels_path=bst_atlas_labels_path)
    generate_resection_overlay_plots(preop_mri, postop_mri)

if __name__ == "__main__":
    preop_mri = "/home1/ajoshi/Downloads/example_dataset_resection/preop.nii.gz"
    postop_mri = "/home1/ajoshi/Downloads/example_dataset_resection/postop.nii.gz"
    my_brainsuite = "/project/ajoshi_27/BrainSuite23a"
    bst_atlas_path="/home1/ajoshi/Downloads/bst_atlases/icbm_bst.nii.gz"
    bst_atlas_labels_path="/home1/ajoshi/Downloads/bst_atlases/icbm_bst.label.nii.gz"
    
    auto_resection_mask(preop_mri, postop_mri, BrainSuitePATH=my_brainsuite,
                        bst_atlas_path=bst_atlas_path,bst_atlas_labels_path=bst_atlas_labels_path)

    print('Done')
