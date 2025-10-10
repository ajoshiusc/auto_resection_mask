import sys
import os
from auto_resection_mask import auto_resection_mask

def validate_temp_directory(temp_dir):
    """
    Validate and resolve temp directory path with automatic path resolution.
    Converts relative paths to absolute paths and performs comprehensive validation.
    
    Args:
        temp_dir (str): Directory path (can be relative or absolute)
        
    Returns:
        str: Validated absolute path
        
    Raises:
        SystemExit: If directory is invalid or inaccessible
    """
    original_path = temp_dir
    
    # Convert relative to absolute path
    if not os.path.isabs(temp_dir):
        temp_dir = os.path.abspath(temp_dir)
        print(f"Resolved relative path '{original_path}' to '{temp_dir}'")
    
    # Normalize path for the current platform
    temp_dir = os.path.normpath(temp_dir)
    
    # Check if directory exists
    if not os.path.exists(temp_dir):
        print(f"Error: Directory '{temp_dir}' does not exist")
        print("Please create the directory first or provide an existing directory")
        sys.exit(1)
    
    # Check if it's actually a directory
    if not os.path.isdir(temp_dir):
        print(f"Error: '{temp_dir}' is not a directory")
        sys.exit(1)
    
    # Test write access
    try:
        test_file = os.path.join(temp_dir, 'test_write_access.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"Validated temp directory: {temp_dir}")
    except (PermissionError, OSError) as e:
        print(f"Error: Cannot write to directory '{temp_dir}': {e}")
        print("Please check directory permissions")
        sys.exit(1)
    
    return temp_dir

def main():
    # Check for custom temp directory argument (3rd argument)
    if len(sys.argv) >= 4:
        custom_temp_dir = sys.argv[3]
        print(f"Custom temp directory requested: {custom_temp_dir}")
        
        # Validate and resolve the temp directory
        validated_temp_dir = validate_temp_directory(custom_temp_dir)
        
        # Set environment variables that control PyInstaller extraction
        # Note: These will only affect child processes, not current extraction
        os.environ['TEMP'] = validated_temp_dir
        os.environ['TMP'] = validated_temp_dir
        os.environ['TMPDIR'] = validated_temp_dir
        
        print(f"Set temp environment variables to: {validated_temp_dir}")
        print("Note: Current extraction location is already determined.")
        print("      Custom temp dir will apply to future PyInstaller runs.")
    
    # Standard argument validation for the main function
    if len(sys.argv) < 3:
        print("Usage: resection_identification_core.exe preop_mri postop_mri [temp_dir]")
        print("  preop_mri: Path to pre-operative MRI file")
        print("  postop_mri: Path to post-operative MRI file")
        print("  temp_dir: (Optional) Custom temporary directory for PyInstaller extraction")
        print("           Can be relative (e.g., './temp', 'data') or absolute path")
        print("           Relative paths will be automatically resolved to absolute paths")
        print("")
        print("Examples:")
        print("  resection_identification_core.exe input1.nii.gz input2.nii.gz")
        print("  resection_identification_core.exe input1.nii.gz input2.nii.gz ./temp")
        print("  resection_identification_core.exe input1.nii.gz input2.nii.gz C:\\MyTemp")
        print("")
        print("Note: This is the core executable. Use resection_identification.bat")
        print("      for better temp directory control (sets environment before extraction).")
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