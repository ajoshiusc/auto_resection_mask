"""
This module processes MRI images to delineate resection areas and generate overlay plots.
"""

from autoresec import delineate_resection_pre, delineate_resection_post

from resection_overlay_plots import generate_resection_overlay_plots

preop_mri = (
    "/deneb_disk/auto_resection/seizure_free_patients_from_ken/"
    "TLY_mesial_26_sub_6_01_2024/F1950N3V/sub-F1950N3V-F1950N3V_MRI.nii.gz"
)
postop_mri = (
    "/deneb_disk/auto_resection/seizure_free_patients_from_ken/"
    "TLY_mesial_26_sub_6_01_2024/F1950N3V/sub-F1950N3V-F1950N3V_postRS_MRI.nii.gz"
)
delineate_resection_pre(preop_mri, postop_mri)
delineate_resection_post(preop_mri, postop_mri)


delineate_resection_post(preop_mri, postop_mri)
delineate_resection_pre(preop_mri, postop_mri)

generate_resection_overlay_plots(preop_mri, postop_mri)

print("Done\n")
