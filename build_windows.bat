:: =============================================================================
:: Auto Resection Mask Windows Build Script
:: =============================================================================
:: This script creates a Windows executable for the auto resection mask tool
:: using PyInstaller. It handles conda environment setup, dependency management,
:: and PyTorch CUDA installation for optimal performance.
:: =============================================================================
:: Created with Claude Sonnet 4 in Visual Studio Code
:: Modified by Chinmay Chinara, 2025
:: =============================================================================

@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: CONDA ENVIRONMENT SETUP
:: =============================================================================

REM Create conda environment from requirements file if it doesn't exist
echo Checking conda environment...
conda env list | findstr /C:"autoresec" > nul
if errorlevel 1 (
    REM Environment doesn't exist - create new one
    echo Creating conda environment 'autoresec'...
    call conda create -n autoresec python=3.10 -y
    if errorlevel 1 (
        echo Failed to create conda environment!
        exit /b 1
    )
    
    REM Activate the new environment
    call conda activate autoresec
    
    REM Install base requirements from file
    echo Installing requirements...
    pip install -r requirements_autoresec.txt
    if errorlevel 1 (
        echo Failed to install requirements!
        exit /b 1
    )
    
    REM Install additional packages for PyInstaller compatibility
    echo Installing additional packages for PyInstaller compatibility...
    pip install pandas pytz python-dateutil numpy scipy matplotlib pillow
    if errorlevel 1 (
        echo Failed to install additional packages!
        exit /b 1
    )
    
    REM Install PyTorch with CUDA support for GPU acceleration
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo Failed to install PyTorch with CUDA! Trying CUDA 11.8...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        if errorlevel 1 (
            echo Failed to install PyTorch with CUDA 11.8! Using default PyTorch...
        )
    )
) else (
    REM Environment exists - activate and ensure CUDA PyTorch is installed
    echo Activating existing environment 'autoresec'...
    call conda activate autoresec
    
    REM Upgrade to ensure PyTorch with CUDA support is available
    echo Ensuring PyTorch with CUDA support is installed...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade
    if errorlevel 1 (
        echo Failed to upgrade to CUDA PyTorch! Trying CUDA 11.8...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
        if errorlevel 1 (
            echo Warning: Could not install CUDA PyTorch, using existing version...
        )
    )
)

:: =============================================================================
:: FILE VALIDATION
:: =============================================================================
REM Check for required ICBM brain atlas files
if not exist "icbm_bst.nii.gz" (
    echo Error: icbm_bst.nii.gz not found!
    exit /b 1
)
if not exist "icbm_bst.label.nii.gz" (
    echo Error: icbm_bst.label.nii.gz not found!
    exit /b 1
)

:: =============================================================================
:: BUILD PREPARATION
:: =============================================================================
REM Clean up previous build files to ensure fresh build
echo Cleaning up previous build files...
if exist "dist" rmdir /s /q dist
if exist "build" rmdir /s /q build
if exist "*.spec" del /f /q *.spec

REM Create working directories
mkdir dist 2>nul
mkdir build 2>nul

REM Copy ICBM atlas files to build directory for bundling
echo Copying ICBM template files...
copy /Y "icbm_bst.nii.gz" "build\" >nul
copy /Y "icbm_bst.label.nii.gz" "build\" >nul

:: =============================================================================
:: DEPENDENCY VERIFICATION
:: =============================================================================
REM Verify pandas is available for data processing
echo Verifying pandas installation...
python -c "import pandas; print('pandas version:', pandas.__version__)"
if errorlevel 1 (
    echo Error: pandas not found in current environment!
    echo Installing pandas...
    pip install pandas
    if errorlevel 1 (
        echo Failed to install pandas!
        exit /b 1
    )
)

REM Verify PyTorch CUDA installation for GPU acceleration
echo Verifying PyTorch CUDA installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
if errorlevel 1 (
    echo Warning: Could not verify PyTorch installation!
)

:: =============================================================================
:: PYINSTALLER SPEC FILE GENERATION
:: =============================================================================
REM Create PyInstaller spec file with comprehensive dependency collection
echo Creating PyInstaller spec file...
(
echo # =============================================================================
echo # PyInstaller Spec File for Auto Resection Mask
echo # =============================================================================
echo # This spec file defines how PyInstaller should bundle the application
echo # including all dependencies, data files, and configuration options.
echo # =============================================================================
echo.
echo # -*- mode: python -*-
echo # -*- coding: utf-8 -*-
echo.
echo import os
echo import sys
echo from pathlib import Path
echo.
echo # Try to import PyInstaller utilities for automatic dependency collection
echo try:
echo     from PyInstaller.utils.hooks import collect_all, collect_submodules
echo except ImportError:
echo     print("Warning: PyInstaller hooks not available"^)
echo     def collect_all(package^):
echo         return [], [], []
echo     def collect_submodules(package^):
echo         return []
echo.
echo # List of critical packages to automatically collect dependencies from
echo packages_to_collect = ['pandas', 'nilearn', 'nibabel', 'sklearn', 'scipy', 'numpy', 'matplotlib', 'torch']
echo all_datas = []
echo all_binaries = []
echo all_hiddenimports = []
echo.
echo # Collect dependencies for each package automatically
echo for package in packages_to_collect:
echo     try:
echo         __import__(package^)
echo         # Special handling for torch to avoid DLL issues during collection
echo         if package == 'torch':
echo             print(f'Skipping automatic collection for {package} due to DLL issues'^)
echo             continue
echo         datas, binaries, hiddenimports = collect_all(package^)
echo         all_datas.extend(datas^)
echo         all_binaries.extend(binaries^)
echo         all_hiddenimports.extend(hiddenimports^)
echo         print(f'Successfully collected {package}'^)
echo     except (ImportError, Exception^) as e:
echo         print(f'Warning: Could not collect {package}: {e}'^)
echo         # Add basic hiddenimports even if automatic collection fails
echo         if package == 'pandas':
echo             all_hiddenimports.extend(['pandas', 'pandas.core', 'pandas._libs']^)
echo         elif package == 'nilearn':
echo             all_hiddenimports.extend(['nilearn', 'nilearn.signal', 'nilearn.image']^)
echo         elif package == 'sklearn':
echo             all_hiddenimports.extend(['sklearn', 'sklearn.utils']^)
echo.
echo # Add comprehensive manual hiddenimports for critical modules
echo # These ensure that all necessary submodules are included in the executable
echo manual_hiddenimports = [
echo     # NumPy core modules
echo     'numpy', 'numpy.core', 'numpy.core._multiarray_umath',
echo     # SciPy modules for scientific computing
echo     'scipy', 'scipy.sparse', 'scipy.sparse.linalg', 'scipy.ndimage',
echo     # PyTorch modules for deep learning
echo     'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim',
echo     'torch._C', 'torch.autograd', 'torch.jit',
echo     # Pandas modules for data manipulation
echo     'pandas', 'pandas.core', 'pandas._libs', 'pandas._libs.lib',
echo     'pandas._libs.tslib', 'pandas._libs.hashtable', 'pandas._libs.algos',
echo     # Medical imaging libraries
echo     'monai', 'torchio', 'SimpleITK',
echo     # Neuroimaging library
echo     'nilearn', 'nilearn.signal', 'nilearn.image', 'nilearn.image.image',
echo     # Plotting library
echo     'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends',
echo     'matplotlib.backends.backend_tkagg',
echo     # Machine learning library
echo     'sklearn', 'sklearn.utils', 'sklearn.utils.validation',
echo     # Image processing
echo     'PIL', 'PIL.Image', 'PIL.ImageDraw',
echo     # NIfTI file handling
echo     'nibabel', 'nibabel.nifti1', 'nibabel.analyze',
echo     # Parallel processing
echo     'joblib', 'joblib.parallel'
echo ]
echo all_hiddenimports.extend(manual_hiddenimports^)
echo # Remove duplicates from hiddenimports list
echo all_hiddenimports = list(set(all_hiddenimports^)^)
echo.
echo # Verify ICBM brain atlas files exist before proceeding
echo icbm_files = [
echo     Path('icbm_bst.nii.gz'^),
echo     Path('icbm_bst.label.nii.gz'^)
echo ]
echo.
echo for icbm_file in icbm_files:
echo     if not icbm_file.exists(^):
echo         raise FileNotFoundError(f'Required file {icbm_file} not found'^)
echo.
echo # No encryption for the executable
echo block_cipher = None
echo.
echo # Main Analysis configuration
echo a = Analysis(
echo     ['standalone_wrapper.py'],          # Entry point script
echo     pathex=['.'],                       # Additional paths to search
echo     binaries=all_binaries,              # Collected binary files
echo     datas=[                            # Data files to bundle
echo         ('icbm_bst.nii.gz', '.'^),           # Brain atlas template
echo         ('icbm_bst.label.nii.gz', '.'^)      # Brain atlas labels
echo     ] + all_datas,                     # Plus automatically collected data
echo     hiddenimports=all_hiddenimports,    # Modules to import at runtime
echo     hookspath=[],                       # No custom hooks directory
echo     hooksconfig={},                     # No hook configuration
echo     runtime_hooks=[],                   # No runtime hooks
echo     excludes=[],                        # No modules to exclude
echo     win_no_prefer_redirects=False,      # Windows-specific option
echo     win_private_assemblies=False,       # Windows-specific option
echo     cipher=block_cipher,                # No encryption
echo     noarchive=False                     # Create archive
echo ^)
echo.
echo # Create Python archive
echo pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher^)
echo.
echo # Create final executable
echo exe = EXE(
echo     pyz,                                # Python archive
echo     a.scripts,                          # Script files
echo     a.binaries,                         # Binary dependencies
echo     a.zipfiles,                         # ZIP files
echo     a.datas,                            # Data files
echo     [],                                 # Additional files
echo     name='auto_resection_mask_win',     # Executable name
echo     debug=False,                        # No debug mode
echo     bootloader_ignore_signals=False,    # Handle signals normally
echo     strip=False,                        # Don't strip symbols
echo     upx=True,                           # Enable UPX compression
echo     upx_exclude=['torch*', 'c10*', '*torch*', '*.dll'],  # Exclude from compression
echo     runtime_tmpdir=None,                # Default temp directory
echo     console=True,                       # Console application
echo     disable_windowed_traceback=False,   # Show tracebacks
echo     target_arch=None,                   # Auto-detect architecture
echo     codesign_identity=None,             # No code signing
echo     entitlements_file=None              # No entitlements
echo ^)
) > auto_resection_mask_win.spec

:: =============================================================================
:: PYINSTALLER INSTALLATION AND EXECUTION
:: =============================================================================
REM Install PyInstaller if not already available
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo Failed to install PyInstaller!
        exit /b 1
    )
)

REM Build the executable using the generated spec file
echo Building executable with PyInstaller...
pyinstaller auto_resection_mask_win.spec --clean
if errorlevel 1 (
    echo Build failed! Check the error messages above.
    exit /b 1
)

:: =============================================================================
:: BUILD VERIFICATION AND COMPLETION
:: =============================================================================
REM Verify the executable was successfully created
if not exist "dist\auto_resection_mask_win.exe" (
    echo Error: Executable was not created!
    exit /b 1
)

REM Additional verification step
echo Verifying ICBM files are bundled correctly...

:: =============================================================================
:: BUILD COMPLETION SUMMARY
:: =============================================================================
echo.
echo Build completed successfully!
echo The executable is available at: dist\auto_resection_mask_win.exe
echo.
echo Usage: auto_resection_mask_win.exe preop_mri postop_mri
echo - preop_mri: Path to pre-operative MRI file
echo - postop_mri: Path to post-operative MRI file
echo.
echo Note: ICBM files are bundled with the executable
echo.
echo =============================================================================
echo Build process completed. The executable includes:
echo - PyTorch with CUDA support for GPU acceleration
echo - All required Python dependencies
echo - ICBM brain atlas files
echo - Comprehensive error handling
echo =============================================================================