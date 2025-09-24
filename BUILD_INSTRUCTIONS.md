# Building on Windows, Linux and Mac (Apple Silicon)

To build the `auto_resection_mask` project, ensure the following files are present in your project directory. Also make sure you clone the `brainstorm-plugin` branch to work on.

## Directory Structure

```
auto_resection_mask-main/
├── standalone_wrapper.py
├── auto_resection_mask.py
├── requirements_autoresec.txt
├── requirements_autoresec_mac.txt
├── build_windows.bat
├── build_mac.sh
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
- `standalone_wrapper.py` — Entry point wrapper script
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
- `build_mac.sh` — Mac build script
- `build_linux.sh` — Linux build script

## Build Instructions

1. **Place all files in the same directory:**  
   Ensure all listed files are located directly within the `auto_resection_mask-brainstorm-plugin` directory.

2. **Install Conda:**  
   Make sure Conda is installed on your system before proceeding. Install from: https://www.anaconda.com/download

3. **Build the executable:**  
   Execute the build script:
   ```bash
   # Windows
   build_windows.bat

   # Mac
   chmod +x build_mac.sh
   ./build_mac.sh
   
   # Linux
   chmod +x build_linux.sh
   ./build_linux.sh
   ```
   This process will bundle all files and dependencies into a single executable under the folder ```dist``` in the same directory.

4. **Test the build:**
   Execute:
   ```bash
   # Windows
   dist\auto_resection_mask_win.exe

   # Mac
   ./dist/auto_resection_mask_mac

   # Linux
   ./dist/auto_resection_mask_linux
   ```
   If you get the response as under then the binary was successsully built
   ```bash
   Usage: auto_resection_mask_xxx preop_mri postop_mri brainsuite_path
     preop_mri: Path to pre-operative MRI file
     postop_mri: Path to post-operative MRI file
     brainsuite_path: Path to BrainSuite installation
   ```
---
**Note:**  
- If you encounter permission errors, double-check the executable permissions for `build_xxx` files.
- For troubleshooting Conda installation, refer to the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
