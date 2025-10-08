@echo off
setlocal enabledelayedexpansion

:: Auto Resection Mask - PyInstaller Extraction Controller
:: This script controls where PyInstaller extracts the _MEIxxxx folder

:: Check number of arguments
if "%~3"=="" goto run_default
if not "%~4"=="" goto show_usage

:: Custom temp directory specified
set "CUSTOM_TEMP=%~3"

:: Convert relative path to absolute path
pushd "!CUSTOM_TEMP!" 2>nul
if errorlevel 1 (
    echo Error: Cannot access directory '!CUSTOM_TEMP!'
    echo Please ensure the directory exists and you have access to it.
    exit /b 1
)
set "CUSTOM_TEMP=%CD%"
popd

echo Resolved temp directory to: !CUSTOM_TEMP!

:: Ensure path is properly formatted for Windows and Qt
set "CUSTOM_TEMP=!CUSTOM_TEMP:/=\!"

:: Final validation - ensure it's an absolute path
echo "!CUSTOM_TEMP!" | findstr "^[A-Za-z]:\\" >nul
if errorlevel 1 (
    echo Error: Must provide an absolute Windows path (e.g., C:\MyTemp or D:\MyFolder)
    echo Received: !CUSTOM_TEMP!
    exit /b 1
)

echo Using absolute temp directory: !CUSTOM_TEMP!

:: Test write access
echo test > "!CUSTOM_TEMP!\test_write.tmp" 2>nul
if errorlevel 1 (
    echo Error: Custom temp directory '!CUSTOM_TEMP!' is not writable
    exit /b 1
)
del "!CUSTOM_TEMP!\test_write.tmp" 2>nul

:: Set environment variables BEFORE starting PyInstaller executable
set "TEMP=!CUSTOM_TEMP!"
set "TMP=!CUSTOM_TEMP!"
set "TMPDIR=!CUSTOM_TEMP!"

echo PyInstaller will extract to: !CUSTOM_TEMP!\_MEIxxxxxx
echo Starting application...

:: Run the PyInstaller executable with environment set
"%~dp0auto_resection_mask_core.exe" %1 %2
goto end

:run_default
:: No custom temp directory - run normally  
"%~dp0auto_resection_mask_core.exe" %1 %2
goto end

:show_usage
echo Usage: %~nx0 preop_mri postop_mri [temp_dir]
echo   preop_mri: Path to pre-operative MRI file
echo   postop_mri: Path to post-operative MRI file  
echo   temp_dir: (Optional) Directory for PyInstaller _MEIxxxx extraction
echo            Must be an absolute path (e.g., C:\MyTemp, D:\MyFolder)
echo            Can also be a relative path that will be converted to absolute
echo.
echo Examples:
echo   %~nx0 input1.nii.gz input2.nii.gz
echo   %~nx0 input1.nii.gz input2.nii.gz "C:\MyTempFolder"
echo   %~nx0 input1.nii.gz input2.nii.gz "data"
exit /b 1

:end