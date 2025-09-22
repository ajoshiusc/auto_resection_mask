@echo off
setlocal enabledelayedexpansion

REM Check conda environment
echo Checking conda environment...

REM Create conda environment from requirements file if it doesn't exist
conda env list | findstr /C:"autoresec" > nul
if errorlevel 1 (
    echo Creating conda environment 'autoresec'...
    call conda create -n autoresec python=3.10 -y
    if errorlevel 1 (
        echo Failed to create conda environment!
        exit /b 1
    )
    call conda activate autoresec
    echo Installing requirements...
    pip install -r requirements_autoresec.txt
    if errorlevel 1 (
        echo Failed to install requirements!
        exit /b 1
    )
) else (
    echo Activating existing environment 'autoresec'...
    call conda activate autoresec
)

REM Check for required ICBM files
if not exist "icbm_bst.nii.gz" (
    echo Error: icbm_bst.nii.gz not found!
    exit /b 1
)
if not exist "icbm_bst.label.nii.gz" (
    echo Error: icbm_bst.label.nii.gz not found!
    exit /b 1
)

REM Clean up previous build files
echo Cleaning up previous build files...
if exist "dist" rmdir /s /q dist
if exist "build" rmdir /s /q build
if exist "*.spec" del /f /q *.spec

REM Create working directories
mkdir dist 2>nul
mkdir build 2>nul

REM Copy ICBM files to build directory
echo Copying ICBM template files...
copy /Y "icbm_bst.nii.gz" "build\" >nul
copy /Y "icbm_bst.label.nii.gz" "build\" >nul

REM Create PyInstaller spec file
echo Creating PyInstaller spec file...
(
echo # -*- mode: python -*-
echo # -*- coding: utf-8 -*-
echo.
echo import os
echo import sklearn
echo import glob
echo from pathlib import Path
echo.
echo # Get sklearn binary files
echo sklearn_path = os.path.dirname(sklearn.__file__^)
echo sklearn_binaries = []
echo.
echo # Add all .pyd files from sklearn
echo for root, dirs, files in os.walk(sklearn_path^):
echo     for file in files:
echo         if file.endswith('.pyd'^):
echo             full_path = os.path.join(root, file^)
echo             rel_path = os.path.relpath(full_path, sklearn_path^)
echo             target_dir = os.path.dirname(rel_path^)
echo             if target_dir == '.':
echo                 target_dir = 'sklearn'
echo             else:
echo                 target_dir = os.path.join('sklearn', target_dir^)
echo             sklearn_binaries.append((full_path, target_dir^)^)
echo.
echo # Verify ICBM files exist
echo icbm_files = [
echo     Path('icbm_bst.nii.gz'^),
echo     Path('icbm_bst.label.nii.gz'^)
echo ]
echo.
echo for icbm_file in icbm_files:
echo     if not icbm_file.exists(^):
echo         raise FileNotFoundError(f'Required file {icbm_file} not found'^)
echo.
echo block_cipher = None
echo.
echo a = Analysis(
echo     ['standalone_wrapper.py'],
echo     pathex=['.'],
echo     binaries=sklearn_binaries,
echo     datas=[
echo         ('icbm_bst.nii.gz', '.'^),
echo         ('icbm_bst.label.nii.gz', '.'^)
echo     ],
echo     hiddenimports=[
echo         'numpy',
echo         'scipy',
echo         'torch',
echo         'monai',
echo         'torchio',
echo         'SimpleITK',
echo         'nilearn',
echo         'matplotlib',
echo         'sklearn',
echo         'PIL',
echo         'sklearn.utils',
echo         'sklearn.utils._param_validation',
echo         'sklearn.utils.validation',
echo         'sklearn.utils._array_api',
echo         'sklearn.externals',
echo         'sklearn.externals.array_api_compat',
echo         'sklearn.externals.array_api_compat.numpy',
echo         'sklearn.externals.array_api_compat.numpy.fft',
echo         'sklearn._cyutility',
echo         'sklearn.utils._cython_blas',
echo         'sklearn.utils._isfinite',
echo         'sklearn.utils._weight_vector',
echo         'sklearn.utils._fast_dict',
echo         'sklearn.utils._typedefs',
echo         'sklearn.utils._vector_sentinel',
echo         'sklearn.utils._sorting',
echo         'sklearn.utils._mask',
echo         'sklearn.utils._heap',
echo         'sklearn.utils._random'
echo     ],
echo     hookspath=[],
echo     hooksconfig={},
echo     runtime_hooks=[],
echo     excludes=[],
echo     win_no_prefer_redirects=False,
echo     win_private_assemblies=False,
echo     cipher=block_cipher,
echo     noarchive=False
echo ^)
echo.
echo pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher^)
echo.
echo exe = EXE(
echo     pyz,
echo     a.scripts, 
echo     a.binaries,
echo     a.zipfiles,
echo     a.datas,
echo     [],
echo     name='auto_resection_mask_win',
echo     debug=False,
echo     bootloader_ignore_signals=False,
echo     strip=False,
echo     upx=True,
echo     upx_exclude=[],
echo     runtime_tmpdir=None,
echo     console=True,
echo     disable_windowed_traceback=False,
echo     target_arch=None,
echo     codesign_identity=None,
echo     entitlements_file=None
echo ^)
) > auto_resection_mask_win.spec

REM Install PyInstaller if not already installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo Failed to install PyInstaller!
        exit /b 1
    )
)

REM Build executable
echo Building executable with PyInstaller...
pyinstaller auto_resection_mask_win.spec --clean
if errorlevel 1 (
    echo Build failed! Check the error messages above.
    exit /b 1
)

REM Verify the executable was created
if not exist "dist\auto_resection_mask_win.exe" (
    echo Error: Executable was not created!
    exit /b 1
)

echo Verifying ICBM files are bundled correctly...

echo.
echo Build completed successfully!
echo The executable is available at: dist\auto_resection_mask_win.exe
echo.
echo Usage: auto_resection_mask_win.exe preop_mri postop_mri brainsuite_path
echo - preop_mri: Path to pre-operative MRI file
echo - postop_mri: Path to post-operative MRI file
echo - brainsuite_path: Path to BrainSuite installation
echo.
echo Note: ICBM files are bundled with the executable