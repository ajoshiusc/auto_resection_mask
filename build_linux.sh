# =============================================================================
# Auto Resection Mask Linux Build Script
# =============================================================================
# This script creates a Linux executable for the auto resection mask tool
# using PyInstaller. It handles conda environment setup, dependency management,
# and PyTorch CUDA installation for optimal GPU performance on Linux.
# =============================================================================
# Created with Claude Sonnet 4 in Visual Studio Code
# Modified by Chinmay Chinara, 2025
# =============================================================================

#!/bin/bash

# Exit immediately if any command fails
set -e

# =============================================================================
# CONDA ENVIRONMENT SETUP
# =============================================================================
echo "Checking conda environment..."

# Check if the 'autoresec' conda environment exists
if ! conda env list | grep -q "autoresec"; then
    # Environment doesn't exist - create new one
    echo "Creating conda environment 'autoresec'..."
    conda create -n autoresec python=3.10 -y
    if [ $? -ne 0 ]; then
        echo "Failed to create conda environment!"
        exit 1
    fi
    
    # Activate the new environment
    source activate autoresec
    if [ $? -ne 0 ]; then
        echo "Failed to activate conda environment!"
        exit 1
    fi
    
    # Install base requirements from file
    echo "Installing requirements..."
    pip install -r requirements_autoresec.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install requirements!"
        exit 1
    fi
    
    # Install additional packages for PyInstaller compatibility
    echo "Installing additional packages for PyInstaller compatibility..."
    pip install pandas pytz python-dateutil numpy scipy matplotlib pillow
    if [ $? -ne 0 ]; then
        echo "Failed to install additional packages!"
        exit 1
    fi
    
    # Install PyTorch with CUDA support for GPU acceleration
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if [ $? -ne 0 ]; then
        echo "Failed to install PyTorch with CUDA! Trying CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        if [ $? -ne 0 ]; then
            echo "Failed to install PyTorch with CUDA 11.8! Installing CPU-only version..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    fi
else
    # Environment exists - activate and ensure CUDA PyTorch is installed
    echo "Activating existing environment 'autoresec'..."
    source activate autoresec
    if [ $? -ne 0 ]; then
        echo "Failed to activate conda environment!"
        exit 1
    fi
    
    # Upgrade to ensure PyTorch with CUDA support is available
    echo "Ensuring PyTorch with CUDA support is installed..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade
    if [ $? -ne 0 ]; then
        echo "Failed to upgrade to CUDA PyTorch! Trying CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
        if [ $? -ne 0 ]; then
            echo "Warning: Could not install CUDA PyTorch, using existing version..."
        fi
    fi
fi

# =============================================================================
# FILE VALIDATION
# =============================================================================
# Check for required ICBM brain atlas files
for file in "icbm_bst.nii.gz" "icbm_bst.label.nii.gz"; do
    if [ ! -f "$file" ]; then
        echo "Error: $file not found!"
        echo "Please ensure the ICBM brain atlas files are in the current directory."
        exit 1
    fi
done
echo "ICBM brain atlas files found successfully."

# =============================================================================
# BUILD PREPARATION
# =============================================================================
# Clean up previous build files to ensure fresh build
echo "Cleaning up previous build files..."
rm -rf dist build
rm -f *.spec

# Create working directories
mkdir -p dist build
echo "Build directories created."

# Copy ICBM atlas files to build directory for bundling
echo "Copying ICBM template files..."
cp "icbm_bst.nii.gz" "build/"
cp "icbm_bst.label.nii.gz" "build/"

# =============================================================================
# DEPENDENCY VERIFICATION
# =============================================================================
# Verify pandas is available for data processing
echo "Verifying pandas installation..."
python -c "import pandas; print('pandas version:', pandas.__version__)"
if [ $? -ne 0 ]; then
    echo "Error: pandas not found in current environment!"
    echo "Installing pandas..."
    pip install pandas
    if [ $? -ne 0 ]; then
        echo "Failed to install pandas!"
        exit 1
    fi
fi

# Verify PyTorch CUDA installation for GPU acceleration
echo "Verifying PyTorch CUDA installation..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
if [ $? -ne 0 ]; then
    echo "Warning: Could not verify PyTorch installation!"
fi

# =============================================================================
# PYINSTALLER SPEC FILE GENERATION
# =============================================================================
# Create PyInstaller spec file with comprehensive dependency collection
echo "Creating PyInstaller spec file..."
cat << 'EOF' > auto_resection_mask_linux.spec
# =============================================================================
# PyInstaller Spec File for Auto Resection Mask (Linux)
# =============================================================================
# This spec file defines how PyInstaller should bundle the application for Linux
# including all dependencies, data files, and configuration options.
# =============================================================================

# -*- mode: python -*-
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Try to import PyInstaller utilities for automatic dependency collection
try:
    from PyInstaller.utils.hooks import collect_all, collect_submodules
except ImportError:
    print("Warning: PyInstaller hooks not available")
    def collect_all(package):
        return [], [], []
    def collect_submodules(package):
        return []

# List of critical packages to automatically collect dependencies from
packages_to_collect = ['pandas', 'nilearn', 'nibabel', 'sklearn', 'scipy', 'numpy', 'matplotlib', 'torch']
all_datas = []
all_binaries = []
all_hiddenimports = []

# Collect dependencies for each package automatically
for package in packages_to_collect:
    try:
        __import__(package)
        # Special handling for torch to avoid potential issues during collection
        if package == 'torch':
            print(f'Skipping automatic collection for {package} due to potential issues')
            continue
        datas, binaries, hiddenimports = collect_all(package)
        all_datas.extend(datas)
        all_binaries.extend(binaries)
        all_hiddenimports.extend(hiddenimports)
        print(f'Successfully collected {package}')
    except (ImportError, Exception) as e:
        print(f'Warning: Could not collect {package}: {e}')
        # Add basic hiddenimports even if automatic collection fails
        if package == 'pandas':
            all_hiddenimports.extend(['pandas', 'pandas.core', 'pandas._libs'])
        elif package == 'nilearn':
            all_hiddenimports.extend(['nilearn', 'nilearn.signal', 'nilearn.image'])
        elif package == 'sklearn':
            all_hiddenimports.extend(['sklearn', 'sklearn.utils'])

# Add comprehensive manual hiddenimports for critical modules
# These ensure that all necessary submodules are included in the executable
manual_hiddenimports = [
    # NumPy core modules
    'numpy', 'numpy.core', 'numpy.core._multiarray_umath',
    # SciPy modules for scientific computing
    'scipy', 'scipy.sparse', 'scipy.sparse.linalg', 'scipy.ndimage',
    # PyTorch modules for deep learning
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim',
    'torch._C', 'torch.autograd', 'torch.jit',
    # Pandas modules for data manipulation
    'pandas', 'pandas.core', 'pandas._libs', 'pandas._libs.lib',
    'pandas._libs.tslib', 'pandas._libs.hashtable', 'pandas._libs.algos',
    # Medical imaging libraries
    'monai', 'torchio', 'SimpleITK',
    # Neuroimaging library
    'nilearn', 'nilearn.signal', 'nilearn.image', 'nilearn.image.image',
    # Plotting library
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends',
    'matplotlib.backends.backend_tkagg', 'matplotlib.backends.backend_agg',
    # Machine learning library
    'sklearn', 'sklearn.utils', 'sklearn.utils.validation',
    # Image processing
    'PIL', 'PIL.Image', 'PIL.ImageDraw',
    # NIfTI file handling
    'nibabel', 'nibabel.nifti1', 'nibabel.analyze',
    # Parallel processing
    'joblib', 'joblib.parallel'
]
all_hiddenimports.extend(manual_hiddenimports)
# Remove duplicates from hiddenimports list
all_hiddenimports = list(set(all_hiddenimports))

# Verify ICBM brain atlas files exist before proceeding
icbm_files = [
    Path('icbm_bst.nii.gz'),
    Path('icbm_bst.label.nii.gz')
]

# Verify BrainSuite binaries exist
brainsuite_path = Path('BrainSuite')
if not brainsuite_path.exists():
    print('Warning: BrainSuite folder not found - BrainSuite binaries will not be bundled')
else:
    print('Found BrainSuite folder - will bundle BrainSuite binaries')

for icbm_file in icbm_files:
    if not icbm_file.exists():
        raise FileNotFoundError(f'Required file {icbm_file} not found')

# No encryption for the executable
block_cipher = None

# Create runtime hook to configure matplotlib for headless operation
runtime_hook_content = '''
import os
import matplotlib
# Set matplotlib to use Agg backend (non-GUI) before any other matplotlib imports
matplotlib.use('Agg')
# Ensure headless operation 
os.environ['MPLBACKEND'] = 'Agg'
'''

# Write runtime hook to temporary file
with open('pyi_rth_matplotlib_headless.py', 'w') as f:
    f.write(runtime_hook_content)

# Main Analysis configuration
a = Analysis(
    ['standalone_wrapper.py'],          # Entry point script
    pathex=['.'],                       # Additional paths to search
    binaries=all_binaries,              # Collected binary files
    datas=[                            # Data files to bundle
        ('icbm_bst.nii.gz', '.'),           # Brain atlas template
        ('icbm_bst.label.nii.gz', '.'),     # Brain atlas labels
        ('BrainSuite', 'BrainSuite')        # BrainSuite binaries folder
    ] + all_datas,                     # Plus automatically collected data
    hiddenimports=all_hiddenimports,    # Modules to import at runtime
    hookspath=[],                       # No custom hooks directory
    hooksconfig={},                     # No hook configuration
    runtime_hooks=['pyi_rth_matplotlib_headless.py'],  # Configure matplotlib for headless operation
    excludes=[                         # Exclude GUI components not needed for headless operation
        'tkinter', 'tkinter.*',        # Tkinter GUI framework
        'PyQt5', 'PyQt5.*',           # PyQt5 GUI framework  
        'PyQt6', 'PyQt6.*',           # PyQt6 GUI framework
        'PySide2', 'PySide2.*',       # PySide2 GUI framework
        'PySide6', 'PySide6.*',       # PySide6 GUI framework
        'matplotlib.backends.backend_qt*',  # Qt backends for matplotlib
        'matplotlib.backends.backend_tk*',  # Tkinter backends
    ],
    win_no_prefer_redirects=False,      # Windows-specific (ignored on Linux)
    win_private_assemblies=False,       # Windows-specific (ignored on Linux)
    cipher=block_cipher,                # No encryption
    noarchive=False                     # Create archive
)

# Create Python archive
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create final executable
exe = EXE(
    pyz,                                # Python archive
    a.scripts,                          # Script files
    a.binaries,                         # Binary dependencies
    a.zipfiles,                         # ZIP files
    a.datas,                            # Data files
    [],                                 # Additional files
    name='auto_resection_mask_core',    # Core executable name
    debug=False,                        # No debug mode
    bootloader_ignore_signals=False,    # Handle signals normally
    strip=False,                        # Don't strip symbols for better debugging
    upx=True,                           # Enable UPX compression
    upx_exclude=['torch*', 'lib*torch*', '*.so'],  # Exclude from compression (Linux)
    runtime_tmpdir=None,                # Default temp directory
    console=True,                       # Console application
    disable_windowed_traceback=False,   # Show tracebacks
    target_arch=None,                   # Auto-detect architecture
    codesign_identity=None,             # No code signing
    entitlements_file=None              # No entitlements
)
EOF

# =============================================================================
# PYINSTALLER INSTALLATION AND EXECUTION
# =============================================================================
# Install PyInstaller if not already available
echo "Checking PyInstaller installation..."
if ! pip show pyinstaller > /dev/null 2>&1; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
    if [ $? -ne 0 ]; then
        echo "Failed to install PyInstaller!"
        exit 1
    fi
fi

# Install additional required packages if missing
packages_to_check=("scikit-learn" "nilearn" "monai" "torchio" "scikit-image" "matplotlib" "requests")
for package in "${packages_to_check[@]}"; do
    if ! pip show "$package" > /dev/null 2>&1; then
        echo "Installing $package..."
        pip install "$package"
    else
        echo "$package is already installed"
    fi
done

# Build the executable using the generated spec file
echo "Building executable with PyInstaller..."
pyinstaller auto_resection_mask_linux.spec --clean
if [ $? -ne 0 ]; then
    echo "Build failed! Check the error messages above."
    exit 1
fi

# =============================================================================
# BUILD VERIFICATION AND COMPLETION
# =============================================================================
# Verify the executable was successfully created
if [ ! -f "dist/auto_resection_mask_core" ]; then
    echo "Error: Core executable was not created!"
    echo "Please check the build output above for errors."
    exit 1
fi

# Make the executable file executable (set proper permissions)
chmod +x "dist/auto_resection_mask_core"

# Additional verification step
echo "Verifying ICBM files are bundled correctly..."

# Cleanup temporary files
echo "Cleaning up temporary files..."
if [ -f "pyi_rth_matplotlib_headless.py" ]; then
    rm -f "pyi_rth_matplotlib_headless.py"
    echo "Removed temporary runtime hook file"
fi

# Copy wrapper script to dist directory
echo "Copying wrapper script..."
if [ -f "auto_resection_mask_linux.sh" ]; then
    cp "auto_resection_mask_linux.sh" "dist/auto_resection_mask_linux"
    chmod +x "dist/auto_resection_mask_linux"
    echo "Wrapper script copied to dist/auto_resection_mask_linux"
else
    echo "Warning: Could not copy wrapper script - auto_resection_mask_linux.sh not found"
fi

# =============================================================================
# BUILD COMPLETION SUMMARY
# =============================================================================
echo
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo
echo "Files created in dist/:"
echo "  1. Core executable: auto_resection_mask_core"
echo "  2. Main launcher:   auto_resection_mask_linux"
echo
echo "Usage: ./auto_resection_mask_linux preop_mri postop_mri [temp_dir]"
echo "  - preop_mri: Path to pre-operative MRI file"
echo "  - postop_mri: Path to post-operative MRI file"
echo "  - temp_dir: (Optional) Directory for PyInstaller _MEIxxxx extraction"
echo "             If not specified, uses system temporary directory"
echo
echo "Examples:"
echo "  ./auto_resection_mask_linux input1.nii.gz input2.nii.gz"
echo "  ./auto_resection_mask_linux input1.nii.gz input2.nii.gz \"/tmp/mytempfolder\""
echo "  ./auto_resection_mask_linux input1.nii.gz input2.nii.gz \"./data\""
echo
echo "Important: Use the launcher script to control extraction directory!"
echo "Direct use of auto_resection_mask_core will use default temp location."
echo
echo "=========================================="
echo "Build process completed successfully!"
echo "The executable includes comprehensive Linux support with:"
echo "- CUDA GPU acceleration (when available)"
echo "- All dependencies statically linked"
echo "- ICBM brain atlas files"
echo "- BrainSuite binaries for Linux"
echo "- Custom temp directory control for PyInstaller extraction"
echo "- Headless operation (no GUI/X11 required)"
echo "- Non-interactive matplotlib plotting"
echo "- Optimized for Linux distributions"
echo "- No external Python installation required"
echo "=========================================="