#!/bin/bash

# Exit on any error
set -e

# Function to check and install packages
check_and_install_package() {
    if ! pip show "$1" > /dev/null 2>&1; then
        echo "Installing $1..."
        pip install "$1"
    fi
}

echo "Checking conda environment..."

# Check if conda environment exists
if ! conda env list | grep -q "autoresec"; then
    echo "Creating conda environment 'autoresec'..."
    conda create -n autoresec python=3.10 -y || exit 1
    source activate autoresec || exit 1
    echo "Installing requirements..."
    pip install -r requirements_autoresec.txt || exit 1
else
    echo "Activating existing environment 'autoresec'..."
    source activate autoresec || exit 1
fi

# Check for required ICBM files
for file in "icbm_bst.nii.gz" "icbm_bst.label.nii.gz"; do
    if [ ! -f "$file" ]; then
        echo "Error: $file not found!"
        exit 1
    fi
done

# Clean up previous build files
echo "Cleaning up previous build files..."
rm -rf dist build
rm -f *.spec

# Create working directories
mkdir -p dist build

# Create spec file
echo "Creating PyInstaller spec file..."
cat << 'EOF' > auto_resection_mask.spec
# -*- mode: python -*-
# -*- coding: utf-8 -*-

import os
import sklearn
from pathlib import Path

# Get sklearn binary files
sklearn_path = os.path.dirname(sklearn.__file__)
sklearn_binaries = []

# Add all shared libraries from sklearn
for root, dirs, files in os.walk(sklearn_path):
    for file in files:
        if file.endswith('.dylib'):  # .dylib files for macOS
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, sklearn_path)
            target_dir = os.path.dirname(rel_path)
            if target_dir == '.':
                target_dir = 'sklearn'
            else:
                target_dir = os.path.join('sklearn', target_dir)
            sklearn_binaries.append((full_path, target_dir))

block_cipher = None

a = Analysis(
    ['standalone_wrapper.py'],
    pathex=['.'],
    binaries=sklearn_binaries,
    datas=[
        ('icbm_bst.nii.gz', '.'),
        ('icbm_bst.label.nii.gz', '.')
    ],
    hiddenimports=[
        'numpy', 'scipy', 'torch', 'monai', 'torchio',
        'SimpleITK', 'nilearn', 'matplotlib', 'sklearn', 'PIL',
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
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='auto_resection_mask_mac',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None
)
EOF

# Install PyInstaller if needed
check_and_install_package "pyinstaller"

# Build executable
echo "Building executable with PyInstaller..."
pyinstaller auto_resection_mask_mac.spec --clean

# Verify the executable was created
if [ ! -f "dist/auto_resection_mask_mac" ]; then
    echo "Error: Executable was not created!"
    exit 1
fi

echo
echo "Build completed successfully!"
echo "The executable is available at: dist/auto_resection_mask_mac"
echo
echo "Usage: ./auto_resection_mask_mac preop_mri postop_mri brainsuite_path"
echo "- preop_mri: Path to pre-operative MRI file"
echo "- postop_mri: Path to post-operative MRI file"
echo "- brainsuite_path: Path to BrainSuite installation"
echo
echo "Note: ICBM files are bundled with the executable"