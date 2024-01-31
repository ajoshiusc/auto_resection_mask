import os
import glob


def main():

    sub_list = glob.glob("/scratch1/ajoshi/temp_4freesurfer/S*")

    for sub in sub_list:
        
        mri_file = sub
        subid = mri_file.removesuffix('.nii.gz')
        #print(subid, mri_file)
        
        cmd = (
            "sbatch freesurfer_recon_all.job " + mri_file + ' ' + subid
        )
        print(cmd)
 
if __name__ == "__main__":
    main()
