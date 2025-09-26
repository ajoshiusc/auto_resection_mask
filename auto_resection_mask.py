# Description: This script is used to generate the resection mask for the preoperative MRI using the postoperative MRI.
from autoresec import delineate_resection_pre, delineate_resection_post
from resection_overlay_plots import generate_resection_overlay_plots



def auto_resection_mask(preop_mri, postop_mri, 
                        bst_atlas_path="bst_atlases/icbm_bst.nii.gz",
                        bst_atlas_labels_path="bst_atlases/icbm_bst.label.nii.gz" ):
    
    delineate_resection_post(preop_mri, postop_mri,
                            bst_atlas_path=bst_atlas_path,bst_atlas_labels_path=bst_atlas_labels_path)
    delineate_resection_pre(preop_mri, postop_mri,
                           bst_atlas_path=bst_atlas_path,bst_atlas_labels_path=bst_atlas_labels_path)
    generate_resection_overlay_plots(preop_mri, postop_mri)

if __name__ == "__main__":
    preop_mri = "data/preop.nii.gz"
    postop_mri = "data/postop.nii.gz"
    bst_atlas_path="icbm_bst.nii.gz"
    bst_atlas_labels_path="icbm_bst.label.nii.gz"
    
    auto_resection_mask(preop_mri, postop_mri,
                        bst_atlas_path=bst_atlas_path,bst_atlas_labels_path=bst_atlas_labels_path)

    print('Done')
