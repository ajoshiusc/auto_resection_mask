# Auto Resection Mask

This repo is used to build a dockerized package for the auto resection mask module.

Here is an [example dataset](https://drive.google.com/drive/folders/1vuI-YwELv8ZMxgxF7ioWN9ceHjArwV91) to run through the module. 

The dockerized package:
https://hub.docker.com/r/chinmaychinara/auto-resection-mask

## Steps to run
1. Make sure you have Docker Desktop installed and running in your machine. For more details check: https://hub.docker.com/

2. <ins>Pull the image</ins>
- For Windows, Linux, Apple intel: `docker pull chinmaychinara/auto-resection-mask:latest`
- For Apple Slicon: `docker pull --platform linux/amd64 chinmaychinara/auto-resection-mask:latest`

3. <ins>Run the image</ins>
- For Windows, Linux (with NVIDIA GPU): `docker run --gpus all /path/to/your/data:/data chinmaychinara/auto-resection-mask:latest /data/preop.nii.gz /data/postop.nii.gz`<br>
  <ins>NOTE:</ins> For GPU usage i.e. for the flag `gpus –all` to work configure [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) in your machine.
- For Windows, Linux, Apple Intel (without NVIDIA GPU): `docker run /path/to/your/data:/data chinmaychinara/auto-resection-mask:latest /data/preop.nii.gz /data/postop.nii.gz`
- For Apple Silicon: `docker run --platform linux/amd64 /path/to/your/data:/data chinmaychinara/auto-resection-mask:latest /data/preop.nii.gz /data/postop.nii.gz`

4. The outputs will be saved in the same folder as the input data if everything runs successfully. For details on how to interpret the results please refer to the [main repo](https://github.com/chinmaychinara91/auto_resection_mask/tree/main).
