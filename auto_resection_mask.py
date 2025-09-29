#!/usr/bin/env python3
"""
Auto resection mask processing for pre-op and post-op MRI scans.
Can be used as a module or as a command-line script.

Usage as CLI: python auto_resection_mask.py <preop_mri> <postop_mri> [options]
Usage as module: from auto_resection_mask import auto_resection_mask
"""

import argparse
import sys
import os

# Description: This script is used to generate the resection mask for the preoperative MRI using the postoperative MRI.
from autoresec import delineate_resection_pre, delineate_resection_post
from resection_overlay_plots import generate_resection_overlay_plots


def auto_resection_mask(preop_mri, postop_mri, BrainSuitePATH,
                        bst_atlas_path="bst_atlases/icbm_bst.nii.gz",
                        bst_atlas_labels_path="bst_atlases/icbm_bst.label.nii.gz"):
    """
    Generate resection masks for pre-op and post-op MRI scans.
    
    Args:
        preop_mri (str): Path to pre-operative MRI scan
        postop_mri (str): Path to post-operative MRI scan  
        BrainSuitePATH (str): Path to BrainSuite installation
        bst_atlas_path (str): Path to BrainSuite atlas
        bst_atlas_labels_path (str): Path to BrainSuite atlas labels
    """
    
    delineate_resection_post(preop_mri, postop_mri, BrainSuitePATH=BrainSuitePATH,
                            bst_atlas_path=bst_atlas_path, bst_atlas_labels_path=bst_atlas_labels_path)
    delineate_resection_pre(preop_mri, postop_mri, BrainSuitePATH=BrainSuitePATH,
                           bst_atlas_path=bst_atlas_path, bst_atlas_labels_path=bst_atlas_labels_path)
    generate_resection_overlay_plots(preop_mri, postop_mri)


def main():
    """Command-line interface for auto_resection_mask processing."""
    parser = argparse.ArgumentParser(
        description='Auto resection mask processing for pre-op and post-op MRI scans',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('preop_mri', 
                       help='Path to pre-operative MRI scan (.nii.gz)')
    parser.add_argument('postop_mri', 
                       help='Path to post-operative MRI scan (.nii.gz)')
    
    # Optional arguments
    parser.add_argument('--brainsuite-path', 
                       default='/home/ajoshi/Software/BrainSuite23a',
                       help='Path to BrainSuite installation')
    parser.add_argument('--bst-atlas-path', 
                       default='bst_atlases/icbm_bst.nii.gz',
                       help='Path to BrainSuite atlas')
    parser.add_argument('--bst-atlas-labels-path', 
                       default='bst_atlases/icbm_bst.label.nii.gz',
                       help='Path to BrainSuite atlas labels')
    parser.add_argument('--subject-id', 
                       help='Subject ID for logging (optional)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.isfile(args.preop_mri):
        print(f"Error: Pre-op MRI file not found: {args.preop_mri}")
        sys.exit(1)
    
    if not os.path.isfile(args.postop_mri):
        print(f"Error: Post-op MRI file not found: {args.postop_mri}")
        sys.exit(1)
    
    # Print processing info
    subject_info = f" (Subject: {args.subject_id})" if args.subject_id else ""
    print(f"Processing auto resection mask{subject_info}...")
    print(f"Pre-op MRI: {args.preop_mri}")
    print(f"Post-op MRI: {args.postop_mri}")
    print(f"BrainSuite path: {args.brainsuite_path}")
    print(f"Atlas path: {args.bst_atlas_path}")
    print(f"Atlas labels path: {args.bst_atlas_labels_path}")
    print("-" * 60)
    
    # Run auto_resection_mask
    auto_resection_mask(
        args.preop_mri,
        args.postop_mri,
        BrainSuitePATH=args.brainsuite_path,
        bst_atlas_path=args.bst_atlas_path,
        bst_atlas_labels_path=args.bst_atlas_labels_path
    )
    
    subject_info = f" {args.subject_id}" if args.subject_id else ""
    print(f"✓ Successfully processed{subject_info}")


if __name__ == "__main__":
    main()
