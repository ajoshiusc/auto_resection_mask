@echo off
setlocal enabledelayedexpansion

:: Resection Identification - Windows Launcher with Automatic Path Resolution
:: This script provides the same functionality as Linux/Mac builds with comprehensive
:: automatic path resolution for custom temp directories.
:: =============================================================================
:: Created with Claude Sonnet 4 in Visual Studio Code
:: Supervised by Chinmay Chinara, 2025
:: =============================================================================

:: Check number of arguments
if "%~3"=="" goto run_default
if not "%~4"=="" goto show_usage

:: Custom temp directory specified - apply automatic path resolution
set "CUSTOM_TEMP=%~3"
set "ORIGINAL_PATH=%~3"

echo Custom temp directory requested: !ORIGINAL_PATH!

:: Step 1: Convert relative path to absolute path automatically
:: Check if path is already absolute (contains drive letter)
set "TEMP_CHECK=!CUSTOM_TEMP:~1,2!"
if "!TEMP_CHECK!"==":\" (
    :: Already absolute path
    echo Using provided absolute path: !CUSTOM_TEMP!
) else (
    :: Not absolute - resolve relative path
    echo Converting relative path to absolute...
    
    :: Handle special relative path cases
    if "!CUSTOM_TEMP!"=="." (
        set "CUSTOM_TEMP=%CD%"
    ) else if "!CUSTOM_TEMP!"==".." (
        pushd ".."
        set "CUSTOM_TEMP=!CD!"
        popd
    ) else (
        :: For other relative paths, resolve from current directory
        pushd "!CUSTOM_TEMP!" 2>nul
        if errorlevel 1 (
            :: Path doesn't exist yet - create absolute path by combining with current dir
            set "CUSTOM_TEMP=%CD%\!CUSTOM_TEMP!"
        ) else (
            :: Path exists - get its absolute path
            set "CUSTOM_TEMP=!CD!"
            popd
        )
    )
    echo Resolved '!ORIGINAL_PATH!' to '!CUSTOM_TEMP!'
)

:: Step 2: Normalize path separators for Windows
set "CUSTOM_TEMP=!CUSTOM_TEMP:/=\!"

:: Step 3: Final path validation
set "FINAL_CHECK=!CUSTOM_TEMP:~1,2!"
if not "!FINAL_CHECK!"==":\" (
    echo Error: Could not resolve to valid absolute Windows path
    echo Original: !ORIGINAL_PATH!
    echo Resolved: !CUSTOM_TEMP!
    echo Please provide a valid directory path
    exit /b 1
)

:: Step 4: Directory existence and access validation
if not exist "!CUSTOM_TEMP!" (
    echo Directory '!CUSTOM_TEMP!' does not exist
    echo Creating directory...
    mkdir "!CUSTOM_TEMP!" 2>nul
    if errorlevel 1 (
        echo Error: Could not create directory '!CUSTOM_TEMP!'
        echo Please check permissions and path validity
        exit /b 1
    )
    echo Created directory: !CUSTOM_TEMP!
)

:: Step 5: Write access test
echo test > "!CUSTOM_TEMP!\test_write.tmp" 2>nul
if errorlevel 1 (
    echo Error: Directory '!CUSTOM_TEMP!' is not writable
    echo Please check directory permissions
    exit /b 1
)
del "!CUSTOM_TEMP!\test_write.tmp" 2>nul

:: Step 6: Set environment variables BEFORE starting PyInstaller executable
set "TEMP=!CUSTOM_TEMP!"
set "TMP=!CUSTOM_TEMP!"
set "TMPDIR=!CUSTOM_TEMP!"

echo ================================================================
echo Automatic Path Resolution Summary:
echo   Original input: !ORIGINAL_PATH!
echo   Resolved to:    !CUSTOM_TEMP!
echo   Directory validated and accessible
echo   PyInstaller will extract to: !CUSTOM_TEMP!\_MEIxxxxxx
echo ================================================================
echo Starting application...

:: Run the PyInstaller executable with environment set
"%~dp0resection_identification_core.exe" %1 %2
goto end

:run_default
:: No custom temp directory - run normally  
"%~dp0resection_identification_core.exe" %1 %2
goto end

:show_usage
echo Usage: %~nx0 preop_mri postop_mri [temp_dir]
echo   preop_mri: Path to pre-operative MRI file
echo   postop_mri: Path to post-operative MRI file  
echo   temp_dir: (Optional) Directory for PyInstaller _MEIxxxx extraction
echo            Supports automatic path resolution:
echo            - Relative paths: "temp", "data", ".", "..", "./my_folder"
echo            - Absolute paths: "C:\MyTemp", "D:\MyFolder"
echo            - Non-existent directories will be created automatically
echo.
echo Automatic Path Resolution Examples:
echo   %~nx0 input1.nii.gz input2.nii.gz                    (default temp)
echo   %~nx0 input1.nii.gz input2.nii.gz "temp"             (./temp)
echo   %~nx0 input1.nii.gz input2.nii.gz "."                (current dir)
echo   %~nx0 input1.nii.gz input2.nii.gz ".."               (parent dir)
echo   %~nx0 input1.nii.gz input2.nii.gz "data/processing"  (./data/processing)
echo   %~nx0 input1.nii.gz input2.nii.gz "C:\MyTempFolder"  (absolute path)
echo.
echo Features:
echo   - Automatic relative-to-absolute path conversion
echo   - Directory creation if needed
echo   - Write permission validation
echo   - Cross-platform path normalization
exit /b 1

:end