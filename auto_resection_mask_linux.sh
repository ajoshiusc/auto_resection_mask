#!/bin/bash

# Auto Resection Mask - PyInstaller Extraction Controller (Linux)
# This script controls where PyInstaller extracts the _MEIxxxx folder

show_usage() {
    echo "Usage: $0 preop_mri postop_mri [temp_dir]"
    echo "  preop_mri: Path to pre-operative MRI file"
    echo "  postop_mri: Path to post-operative MRI file"
    echo "  temp_dir: (Optional) Directory for PyInstaller _MEIxxxx extraction"
    echo "           Can be absolute (/tmp/mytemp) or relative (./data)"
    echo "           Will be converted to absolute path automatically"
    echo ""
    echo "Examples:"
    echo "  $0 input1.nii.gz input2.nii.gz"
    echo "  $0 input1.nii.gz input2.nii.gz \"/tmp/mytempfolder\""
    echo "  $0 input1.nii.gz input2.nii.gz \"./data\""
    exit 1
}

# Check number of arguments
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    show_usage
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If no custom temp directory specified, run normally
if [ $# -eq 2 ]; then
    echo "Using default system temp directory"
    exec "$SCRIPT_DIR/auto_resection_mask_core" "$1" "$2"
fi

# Custom temp directory specified
CUSTOM_TEMP="$3"

# Convert relative path to absolute path
if [[ "$CUSTOM_TEMP" != /* ]]; then
    CUSTOM_TEMP="$(cd "$CUSTOM_TEMP" 2>/dev/null && pwd)"
    if [ $? -ne 0 ]; then
        echo "Error: Cannot access directory '$3'"
        echo "Please ensure the directory exists and you have access to it."
        exit 1
    fi
    echo "Resolved relative path to: $CUSTOM_TEMP"
fi

# Validate custom temp directory exists
if [ ! -d "$CUSTOM_TEMP" ]; then
    echo "Error: Custom temp directory '$CUSTOM_TEMP' does not exist"
    exit 1
fi

# Test write access
if [ ! -w "$CUSTOM_TEMP" ]; then
    echo "Error: Custom temp directory '$CUSTOM_TEMP' is not writable"
    exit 1
fi

# Test write access by creating a temporary file
if ! touch "$CUSTOM_TEMP/test_write.tmp" 2>/dev/null; then
    echo "Error: Cannot write to custom temp directory '$CUSTOM_TEMP'"
    exit 1
fi
rm -f "$CUSTOM_TEMP/test_write.tmp"

echo "Using absolute temp directory: $CUSTOM_TEMP"

# Set environment variables BEFORE starting PyInstaller executable
export TEMP="$CUSTOM_TEMP"
export TMP="$CUSTOM_TEMP"
export TMPDIR="$CUSTOM_TEMP"

echo "PyInstaller will extract to: $CUSTOM_TEMP/_MEIxxxxxx"
echo "Starting application..."

# Run the PyInstaller executable with environment set
exec "$SCRIPT_DIR/auto_resection_mask_core" "$1" "$2"