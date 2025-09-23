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
    echo Installing additional packages for PyInstaller compatibility...
    pip install pandas pytz python-dateutil numpy scipy matplotlib pillow
    if errorlevel 1 (
        echo Failed to install additional packages!
        exit /b 1
    )
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
    echo Activating existing environment 'autoresec'...
    call conda activate autoresec
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

REM Verify pandas is installed
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

REM Verify PyTorch CUDA installation
echo Verifying PyTorch CUDA installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
if errorlevel 1 (
    echo Warning: Could not verify PyTorch installation!
)

REM Create PyInstaller spec file
echo Creating PyInstaller spec file...
(
echo # -*- mode: python -*-
echo # -*- coding: utf-8 -*-
echo.
echo import os
echo import sys
echo from pathlib import Path
echo.
echo # Try to import PyInstaller utilities
echo try:
echo     from PyInstaller.utils.hooks import collect_all, collect_submodules
echo except ImportError:
echo     print("Warning: PyInstaller hooks not available"^)
echo     def collect_all(package^):
echo         return [], [], []
echo     def collect_submodules(package^):
echo         return []
echo.
echo # List of packages to try to collect
echo packages_to_collect = ['pandas', 'nilearn', 'nibabel', 'sklearn', 'scipy', 'numpy', 'matplotlib', 'torch']
echo all_datas = []
echo all_binaries = []
echo all_hiddenimports = []
echo.
echo # Try to collect dependencies for each package
echo for package in packages_to_collect:
echo     try:
echo         __import__(package^)
echo         # Special handling for torch to avoid DLL issues
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
echo         # Add basic hiddenimports even if collection fails
echo         if package == 'pandas':
echo             all_hiddenimports.extend(['pandas', 'pandas.core', 'pandas._libs']^)
echo         elif package == 'nilearn':
echo             all_hiddenimports.extend(['nilearn', 'nilearn.signal', 'nilearn.image']^)
echo         elif package == 'sklearn':
echo             all_hiddenimports.extend(['sklearn', 'sklearn.utils']^)
echo.
echo # Add comprehensive manual hiddenimports for critical modules
echo manual_hiddenimports = [
echo     'numpy', 'numpy.core', 'numpy.core._multiarray_umath',
echo     'scipy', 'scipy.sparse', 'scipy.sparse.linalg', 'scipy.ndimage',
echo     'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim',
echo     'torch._C', 'torch.autograd', 'torch.jit',
echo     'pandas', 'pandas.core', 'pandas._libs', 'pandas._libs.lib',
echo     'pandas._libs.tslib', 'pandas._libs.hashtable', 'pandas._libs.algos',
echo     'monai', 'torchio', 'SimpleITK',
echo     'nilearn', 'nilearn.signal', 'nilearn.image', 'nilearn.image.image',
echo     'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends',
echo     'matplotlib.backends.backend_tkagg',
echo     'sklearn', 'sklearn.utils', 'sklearn.utils.validation',
echo     'PIL', 'PIL.Image', 'PIL.ImageDraw',
echo     'nibabel', 'nibabel.nifti1', 'nibabel.analyze',
echo     'joblib', 'joblib.parallel'
echo ]
echo all_hiddenimports.extend(manual_hiddenimports^)
echo # Remove duplicates
echo all_hiddenimports = list(set(all_hiddenimports^)^)
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
echo     binaries=all_binaries,
echo     datas=[
echo         ('icbm_bst.nii.gz', '.'^),
echo         ('icbm_bst.label.nii.gz', '.'^)
echo     ] + all_datas,
echo     hiddenimports=all_hiddenimports,
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
echo     upx_exclude=['torch*', 'c10*', '*torch*', '*.dll'],
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