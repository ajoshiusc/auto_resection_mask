# auto_resection_mask
Coregisters pre and post-op MRI images and identifies resection.

The code is documented in the jupyter notebook main_automask.ipynb

This notebook shows an example workflow to delineate resection on MRI image.

**Inputs**: Pre-op MRI image, Post-op MRI image  
**Outputs**: Post-op MRI image affinely registered to pre-op MRI image, resection mask (\<subject\>.resection.mask.nii.gz)

To run the code, NVidia GPU is required. 

To run the code, please open main_automask.ipynb as a jupyter notebook, correct paths of our input pre-op and post-op MRIs and run the code.



# Installation
* Make sure you have [BrainSuite](https://brainsuite.org) installed. 
* A devcontainer is available with the repository. Open `.devcontainer.json` file and set the data and code paths.
* Open the cloned folder using vscode. This will prompt you to open the repo in a devcontainer. 
* [Alternatively], you can configure a python environment by manually installing `requirements.txt`
Also, [BrainSuite](HTTP://brainsuite.org) needs to be installed on your system. 

Please contact [Anand A Joshi](ajoshi@usc.edu) if you have any questions.

![image](https://github.com/ajoshiusc/auto_resection_mask/assets/15238551/09d32830-3ae0-4eaa-935e-22e280905dc6)


# Usage

## Demarkating Resection

* Assume as inpot, you have `preop.nii.gz` (pre-op MRI), and `post-op.nii.gz` (post-op MRI with resection). Open Jupyter notebook `main_automask.ipynb` in vscode.
* Go to the **Input Cell** and add the paths to `preop.nii.gz` and `post-op.nii.gz`. Add the path to the BrainSuite installation folder.
* Run the notebook.
  
## [Optional] Importing to BrainStorm

## [Visualization in BrainSuite]
* Open BrainSuite and load *pre-op.nii.gz*
*  






