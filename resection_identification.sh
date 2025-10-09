#!/bin/bash

# Resection Identification - PyInstaller Extraction Controller (Linux)
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
    exec "$SCRIPT_DIR/resection_identification_core" "$1" "$2"
fi

# Custom temp directory specified
CUSTOM_TEMP="$3"

# Convert relative path to absolute path
if [[ "$CUSTOM_TEMP" != /* ]]; then
    # Handle relative paths
    if [ "$CUSTOM_TEMP" = "." ]; then
        CUSTOM_TEMP="$(pwd)"
    elif [[ "$CUSTOM_TEMP" == ./* ]]; then
        CUSTOM_TEMP="$(pwd)/${CUSTOM_TEMP#./}"
    elif [[ "$CUSTOM_TEMP" == ../* ]]; then
        CUSTOM_TEMP="$(cd "$CUSTOM_TEMP" 2>/dev/null && pwd)"
        if [ $? -ne 0 ]; then
            echo "Error: Cannot resolve relative path '$3'"
            exit 1
        fi
    else
        CUSTOM_TEMP="$(pwd)/$CUSTOM_TEMP"
    fi
    echo "Resolved relative path to: $CUSTOM_TEMP"
fi

# Create directory if it doesn't exist
if [ ! -d "$CUSTOM_TEMP" ]; then
    echo "Creating temp directory: $CUSTOM_TEMP"
    if ! mkdir -p "$CUSTOM_TEMP" 2>/dev/null; then
        echo "Error: Could not create directory '$CUSTOM_TEMP'"
        echo "Using default temp directory"
        exec "$SCRIPT_DIR/resection_identification_core" "$1" "$2"
    fi
    echo "Successfully created: $CUSTOM_TEMP"
else
    echo "Using existing temp directory: $CUSTOM_TEMP"
fi

# Test write access by creating a temporary file
if ! touch "$CUSTOM_TEMP/test_write.tmp" 2>/dev/null; then
    echo "Warning: Cannot write to custom temp directory '$CUSTOM_TEMP'"
    echo "Using default temp directory"
    exec "$SCRIPT_DIR/resection_identification_core" "$1" "$2"
fi
rm -f "$CUSTOM_TEMP/test_write.tmp"
echo "Write permission confirmed for: $CUSTOM_TEMP"

# Set environment variables BEFORE starting PyInstaller executable
export TEMP="$CUSTOM_TEMP"
export TMP="$CUSTOM_TEMP"
export TMPDIR="$CUSTOM_TEMP"

echo "PyInstaller will extract to: $CUSTOM_TEMP/_MEIxxxxxx"
echo "Starting resection identification analysis..."

# Run the PyInstaller executable with environment set
exec "$SCRIPT_DIR/resection_identification_core" "$1" "$2"