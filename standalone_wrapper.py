import sys
import os
from auto_resection_mask import auto_resection_mask

def main():
    if len(sys.argv) != 3:
        print("Usage: auto_resection_mask_core.exe preop_mri postop_mri")
        print("  preop_mri: Path to pre-operative MRI file")
        print("  postop_mri: Path to post-operative MRI file")
        print("")
        print("Note: This is the core executable. Use auto_resection_mask_win.bat")
        print("      for custom temp directory support.")
        sys.exit(1)

    preop_mri = sys.argv[1]
    postop_mri = sys.argv[2]
    
    # Get the directory where the bundled files are located
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        application_path = sys._MEIPASS
        
        # Show extraction directory info
        print(f"PyInstaller extracted to: {application_path}")
        current_temp = os.environ.get('TEMP', 'Unknown')
        print(f"Current temp directory: {current_temp}")
    else:
        # Running as script
        application_path = os.path.dirname(os.path.abspath(__file__))
        
    # Set paths for ICBM files that are bundled with the executable
    icbm_template = os.path.join(application_path, 'icbm_bst.nii.gz')
    icbm_label = os.path.join(application_path, 'icbm_bst.label.nii.gz')
    
    # Check if required files exist
    if not all(os.path.exists(f) for f in [preop_mri, postop_mri]):
        print("Error: Input MRI files not found")
        sys.exit(1)
            
    if not all(os.path.exists(f) for f in [icbm_template, icbm_label]):
        print("Error: ICBM template files not found")
        sys.exit(1)

    # Call the main function
    result = auto_resection_mask(
        preop_mri,
        postop_mri,
        icbm_template,
        icbm_label
    )
    
    return result

if __name__ == '__main__':
    main()