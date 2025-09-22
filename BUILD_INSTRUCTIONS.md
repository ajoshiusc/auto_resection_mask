# Building on Windows

To build the `auto_resection_mask` project on Linux, ensure the following files are present in your project directory.

## Directory Structure

```
auto_resection_mask-main/
├── wrapper.py
├── auto_resection_mask.py
├── requirements_autoresec.txt
├── build_linux.sh
├── icbm_bst.nii.gz
├── icbm_bst.label.nii.gz
├── autoresec.py
├── warper.py
├── warp_utils.py
├── aligner.py
├── networks.py
├── deform_losses.py
└── __init__.py
```

## File Description

### Core Files
- `wrapper.py` — Entry point wrapper script
- `auto_resection_mask.py` — Main application code
- `requirements_autoresec.txt` — Package dependencies

### Data Files
- `icbm_bst.nii.gz` — ICBM template file
- `icbm_bst.label.nii.gz` — ICBM label file

### Supporting Python Modules
- `autoresec.py` — Auto resection module
- `warper.py` — Warping utilities
- `warp_utils.py` — Warping utility functions
- `aligner.py` — Alignment functions
- `networks.py` — Network definitions
- `deform_losses.py` — Deformation loss functions

### Build Script
- `build_windows.bat` — Windows build script

## Build Instructions

1. **Place all files in the same directory:**  
   Ensure all listed files are located directly within the `auto_resection_mask-main` directory.

2. **Install Conda:**  
   Make sure Conda is installed on your system before proceeding.

3. **Build the executable:**  
   Execute the build script:
   ```bash
   build_windows.bat
   ```
   This process will bundle all files and dependencies into a single executable under the folder ```dist``` in the same directory.

4. **Test the build:**
   Execute:
   ```bash
   dist\auto_resection_mask_win.exe
   ```
   If you get the response as under then the binary was successsully built
   ```bash
   Usage: auto_resection_mask preop_mri postop_mri brainsuite_path
     preop_mri: Path to pre-operative MRI file
     postop_mri: Path to post-operative MRI file
     brainsuite_path: Path to BrainSuite installation
   ```
---
**Note:**  
- If you encounter permission errors, double-check the executable permissions for `build_windows.bat`.
- For troubleshooting Conda installation, refer to the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
