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
* Open the cloned folder using vscode. This will prompt you to open the repo in a devcontainer. Alternatively, you can configure a python environment by manually installing `requirements.txt`
* [BrainSuite](HTTP://brainsuite.org) needs to be installed on your system. If you don't have it, please [install it](https://brainsuite.org/quickstart/installation).

![image](https://github.com/ajoshiusc/auto_resection_mask/assets/15238551/09d32830-3ae0-4eaa-935e-22e280905dc6)


# Usage

## Running the code

* Assume as inpot, you have `preop.nii.gz` (pre-op MRI), and `post-op.nii.gz` (post-op MRI with resection). Open Jupyter notebook `main_automask.ipynb` in vscode.
* Go to the **Input Cell** and add the paths to `preop.nii.gz` and `post-op.nii.gz`. Add the path to the BrainSuite installation folder.
* Run the notebook.
  
## [Optional] Importing to [BrainStorm](https://neuroimage.usc.edu/brainstorm/Introduction)
The resection mask created by the tool can be imported into BrainStorm during import anatomy.
* Open BrainStorm, Import anatomy as a (BrainSuite processed folder)[https://neuroimage.usc.edu/brainstorm/Tutorials/SegBrainSuite].
* You will see `resection` (resection surface), and `resection_mask` (resection volume). You can visualize them in BrainStorm by right clicking and selecting options of your choice. 

![bstm_resection](https://github.com/ajoshiusc/auto_resection_mask/assets/15238551/4b90cf7a-7ed5-4436-b0dc-b2c5fe7128d6)


## [Optional] Visualization in BrainSuite
* Open BrainSuite and load *pre-op.nii.gz*
* Click on `File->Overlay Volume` and select pre-op.post2pre.nii.gz" and load it. This is the post-op volume coregistred to pre-op volume.
* Click on `File->Open mask volume` and select resection.mask.nii.gz and load it. This will show outline of the identified resection.
* Goto `Tools->Mask Tool` and generate a surface.

<!--- ![buite_resection](https://github.com/ajoshiusc/auto_resection_mask/assets/15238551/dc06a0b2-4ed6-4743-a738-48d51f55cf60) --->

<img src="https://github.com/ajoshiusc/auto_resection_mask/assets/15238551/dc06a0b2-4ed6-4743-a738-48d51f55cf60)" alt="drawing" width="700"/>

  

## Support
* Please contact [Anand A Joshi](ajoshi@usc.edu) if you have any questions.





