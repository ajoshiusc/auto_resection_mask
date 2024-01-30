import os
import glob


def main():

    sub_list = glob.glob("/scratch1/ajoshi/temp_4freesurfer/S*")

    for sub in sub_list:
        #print(sub)
        mri_file = os.path.join(sub,sub[:-11]+'.nii.gz')

        pth, subid = os.path.split(sub)
        #print(subid, mri_file)
        
        cmd = (
            "sbatch freesurfer_recon_all.job " + mri_file + ' ' + subid
        )
        print(cmd)
 
    """
    for sess, n in product((1, 2), range(1, nsub + 1)):
        sub_nii = glob.glob(
            "/scratch1/ajoshi/3T_vs_low_field/3T_mprage_data/subj"
            + str(n)
            + "_vol"
            + str(sess)
            + "/*.nii"
        )[0]
        out_dir = (
            "/scratch1/ajoshi/3T_vs_low_field/3T_mprage_data_BrainSuite/subj"
            + str(n)
            + "_vol"
            + str(sess)
        )

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)


        out_file = out_dir + "/T1.nii.gz"

        if not os.path.isfile(out_file):
            
            img = nib.load(sub_nii)
            data = img.get_fdata()

            p = np.percentile(data, 98)
            print(
                f" max : {data.max()} and min : {data.min()} and 98 percentile: {p}"
            )  # doctest: +SKIP 3237.0
            new_data = data.copy()

            new_data = np.minimum(200.0 * data / p, 255)

            new_img = nib.Nifti1Image(new_data, img.affine, img.header, dtype=np.uint16)

            new_img.to_filename(out_file)
        
        fs_subid = "subj_" + str(n) + "_vol" + str(sess) + "_3T"
        
        cmd = (
            "sbatch freesurfer_recon_all.job " + out_file + ' ' + fs_subid
        )
        print(cmd)
        #cmds_all.append(cmd)
    """

if __name__ == "__main__":
    main()
