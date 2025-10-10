# =============================================================================
# Resection Identification macOS Build Script
# =============================================================================
# This script creates a macOS executable for the resection identification tool
# using PyInstaller. It handles environment setup, dependency management,
# and creates a portable executable with all required libraries bundled.
# =============================================================================
# Created with Claude Sonnet 4 in Visual Studio Code
# Supervised by Chinmay Chinara, 2025
# =============================================================================
# Note: Tested on Apple Silicon only
# =============================================================================

#!/bin/bash

# Exit immediately if any command fails
set -e

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Function to check if a package is installed and install it if missing
check_and_install_package() {
    if ! pip show "$1" > /dev/null 2>&1; then
        echo "Installing $1..."
        pip install "$1"
    else
        echo "$1 is already installed"
    fi
}

# =============================================================================
# CONDA ENVIRONMENT SETUP
# =============================================================================
echo "Checking conda environment..."

# Check if the 'autoresec' conda environment exists
if ! conda env list | grep -q "autoresec"; then
    # Environment doesn't exist - create new one
    echo "Creating conda environment 'autoresec'..."
    conda create -n autoresec python=3.10 -y || exit 1
    
    # Activate the new environment
    source activate autoresec || exit 1
    
    # Install base requirements from macOS-specific requirements file
    echo "Installing requirements..."
    pip install -r requirements_autoresec_mac.txt || exit 1
else
    # Environment exists - just activate it
    echo "Activating existing environment 'autoresec'..."
    source activate autoresec || exit 1
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

# Verify BrainSuite folder exists
if [ -d "BrainSuite" ]; then
    echo "BrainSuite folder found - will be included in build."
else
    echo "Warning: BrainSuite folder not found - will be skipped in build."
fi

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

# =============================================================================
# PYINSTALLER SPEC FILE GENERATION
# =============================================================================
# Create PyInstaller spec file with comprehensive dependency collection
echo "Creating PyInstaller spec file..."
cat << 'EOF' > resection_identification_mac.spec
# =============================================================================
# PyInstaller Spec File for Auto Resection Mask (macOS)
# =============================================================================
# This spec file defines how PyInstaller should bundle the application for macOS
# including all dependencies, data files, and configuration options.
# =============================================================================

# -*- mode: python -*-
# -*- coding: utf-8 -*-

import os
from pathlib import Path

# =============================================================================
# SKLEARN BINARY COLLECTION (macOS-specific)
# =============================================================================
# Try to collect sklearn binary files (.dylib for macOS)
sklearn_binaries = []
try:
    import sklearn
    sklearn_path = os.path.dirname(sklearn.__file__)
    
    # Add all shared libraries from sklearn (macOS uses .dylib files)
    for root, dirs, files in os.walk(sklearn_path):
        for file in files:
            if file.endswith('.dylib'):  # macOS dynamic libraries
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, sklearn_path)
                target_dir = os.path.dirname(rel_path)
                if target_dir == '.':
                    target_dir = 'sklearn'
                else:
                    target_dir = os.path.join('sklearn', target_dir)
                sklearn_binaries.append((full_path, target_dir))
except ImportError:
    print("Warning: sklearn not found, continuing without sklearn binaries")
    sklearn_binaries = []

# No encryption for the executable
block_cipher = None

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

# =============================================================================
# MAIN ANALYSIS CONFIGURATION
# =============================================================================
a = Analysis(
    ['standalone_wrapper.py'],          # Entry point script
    pathex=['.'],                       # Additional paths to search
    binaries=sklearn_binaries,          # Collected sklearn binary files
    datas=[                            # Data files to bundle
        ('icbm_bst.nii.gz', '.'),           # Brain atlas template
        ('icbm_bst.label.nii.gz', '.')      # Brain atlas labels
    ] + ([('BrainSuite', 'BrainSuite')] if brainsuite_path.exists() else []),  # BrainSuite binaries if available
    hiddenimports=[                    # Modules to import at runtime
        # Core scientific computing libraries
        'numpy', 'scipy', 'torch',
        
        # Medical imaging libraries
        'monai', 'torchio', 'SimpleITK', 'nilearn',
        
        # Visualization and plotting
        'matplotlib', 'PIL',
        
        # Machine learning
        'sklearn',
        
        # Image processing
        'skimage', 'skimage.morphology', 'skimage.measure',
        'skimage.segmentation', 'skimage.filters', 'skimage.feature',
        
        # Matplotlib backend modules
        'matplotlib.pyplot', 'matplotlib.backends',
        'matplotlib.backends.backend_tkagg', 'matplotlib.backends.backend_agg',
        'matplotlib.figure', 'matplotlib.axes',
        
        # Networking libraries
        'httpx', 'requests', 'urllib3', 'certifi',
        
        # MONAI submodules for medical imaging
        'monai.utils', 'monai.utils.misc', 'monai.transforms',
        'monai.data', 'monai.networks', 'monai.losses',
        'monai.metrics', 'monai.inferers', 'monai.engines',
        'monai.config', 'monai.config.deviceconfig',
        
        # TorchIO submodules for medical imaging
        'torchio.transforms', 'torchio.data', 'torchio.datasets',
        
        # Nilearn submodules for neuroimaging
        'nilearn.image', 'nilearn.plotting', 'nilearn.masking',
        'nilearn.datasets', 'nilearn._utils', 'nilearn.surface',
        
        # Scikit-learn submodules and utilities
        'sklearn.utils', 'sklearn.utils._param_validation',
        'sklearn.utils.validation', 'sklearn.utils._array_api',
        'sklearn.externals', 'sklearn.externals.array_api_compat',
        'sklearn.externals.array_api_compat.numpy',
        'sklearn.externals.array_api_compat.numpy.fft',
        'sklearn._cyutility', 'sklearn.utils._cython_blas',
        'sklearn.utils._isfinite', 'sklearn.utils._weight_vector',
        'sklearn.utils._fast_dict', 'sklearn.utils._typedefs',
        'sklearn.utils._vector_sentinel', 'sklearn.utils._sorting',
        'sklearn.utils._mask', 'sklearn.utils._heap',
        'sklearn.utils._random'
    ],
    hookspath=[],                       # No custom hooks directory
    hooksconfig={},                     # No hook configuration
    runtime_hooks=[],                   # No runtime hooks
    excludes=[],                        # No modules to exclude
    win_no_prefer_redirects=False,      # Windows-specific (ignored on macOS)
    win_private_assemblies=False,       # Windows-specific (ignored on macOS)
    cipher=block_cipher,                # No encryption
    noarchive=False                     # Create archive
)

# =============================================================================
# PYTHON ARCHIVE CREATION
# =============================================================================
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# =============================================================================
# EXECUTABLE CREATION
# =============================================================================
exe = EXE(
    pyz,                                # Python archive
    a.scripts,                          # Script files
    a.binaries,                         # Binary dependencies
    a.zipfiles,                         # ZIP files
    a.datas,                            # Data files
    name='resection_identification_core',    # Core executable name
    debug=False,                        # No debug mode
    bootloader_ignore_signals=False,    # Handle signals normally
    strip=False,                        # Don't strip symbols for better debugging
    upx=True,                           # Enable UPX compression
    upx_exclude=[],                     # No UPX exclusions (unlike Windows)
    runtime_tmpdir=None,                # Default temp directory
    console=True,                       # Console application
    disable_windowed_traceback=False,   # Show tracebacks
    target_arch=None,                   # Auto-detect architecture
    codesign_identity=None,             # No code signing
    entitlements_file=None              # No entitlements
)
EOF

# =============================================================================
# DEPENDENCY INSTALLATION
# =============================================================================
# Install required packages if not already present
echo "Checking and installing required packages..."

# Install PyInstaller for building the executable
check_and_install_package "pyinstaller"

# Install core scientific computing and machine learning packages
check_and_install_package "scikit-learn"    # Machine learning library
check_and_install_package "nilearn"         # Neuroimaging in Python
check_and_install_package "monai"           # Medical imaging AI toolkit
check_and_install_package "torchio"         # Medical image processing
check_and_install_package "scikit-image"    # Image processing
check_and_install_package "matplotlib"      # Plotting library

# Install networking libraries for data download/upload
check_and_install_package "httpx"           # Modern HTTP client
check_and_install_package "requests"        # HTTP library

# =============================================================================
# EXECUTABLE BUILD PROCESS
# =============================================================================
# Build the executable using the generated spec file
echo "Building executable with PyInstaller..."
pyinstaller resection_identification_mac.spec --clean

# =============================================================================
# BUILD VERIFICATION AND COMPLETION
# =============================================================================
# Verify the executable was successfully created
if [ ! -f "dist/resection_identification_core" ]; then
    echo "Error: Core executable was not created!"
    echo "Please check the build output above for errors."
    exit 1
fi

# Make the executable file executable (set proper permissions)
chmod +x "dist/resection_identification_core"

# Copy wrapper script to dist directory
echo "Copying wrapper script..."
if [ -f "resection_identification.sh" ]; then
    cp "resection_identification.sh" "dist/resection_identification"
    chmod +x "dist/resection_identification"
    echo "Wrapper script copied to dist/resection_identification"
else
    echo "Warning: Could not copy wrapper script - resection_identification.sh not found"
fi

# =============================================================================
# BUILD COMPLETION SUMMARY
# =============================================================================
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo
echo "Files created in dist/:"
echo "  1. Core executable: resection_identification_core"
echo "  2. Main launcher:   resection_identification"
echo
echo "Usage: ./resection_identification preop_mri postop_mri [temp_dir]"
echo "  - preop_mri: Path to pre-operative MRI file"
echo "  - postop_mri: Path to post-operative MRI file"
echo "  - temp_dir: (Optional) Directory for PyInstaller _MEIxxxx extraction"
echo "             If not specified, uses system temporary directory"
echo
echo "Examples:"
echo "  ./resection_identification input1.nii.gz input2.nii.gz"
echo "  ./resection_identification input1.nii.gz input2.nii.gz \"/tmp/mytempfolder\""
echo "  ./resection_identification input1.nii.gz input2.nii.gz \"./data\""
echo
echo "Important: Use the launcher script to control extraction directory!"
echo "Direct use of resection_identification_core will use default temp location."
echo
echo "Features included in this build:"
echo "  - All required Python dependencies bundled"
echo "  - ICBM brain atlas files embedded"
echo "  - BrainSuite binaries included (if available)"
echo "  - Custom temp directory control for PyInstaller extraction"
echo "  - PyTorch for deep learning capabilities"
echo "  - Comprehensive error handling"
echo
echo "=========================================="