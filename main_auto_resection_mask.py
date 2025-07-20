import sys

sys.path.append("/project/ajoshi_27/code_farm/auto_resection_mask")

from auto_resection_mask import auto_resection_mask

preop_mri = "/home1/ajoshi/Downloads/example_dataset_resection/preop.nii.gz"
postop_mri = "/home1/ajoshi/Downloads/example_dataset_resection/postop.nii.gz"
my_brainsuite = "/project/ajoshi_27/BrainSuite23a"
bst_atlas_path="bst_atlases/icbm_bst.nii.gz",
bst_atlas_labels_path="bst_atlases/icbm_bst.label.nii.gz",

auto_resection_mask(preop_mri, postop_mri, BrainSuitePATH=my_brainsuite, bst_atlas_path=bst_atlas_path,bst_atlas_labels_path=bst_atlas_labels_path)

print("Done")
