# Auto Resection Mask

This repo is used to build a dockerized package for the auto resection mask module.

Here is an [example dataset](https://drive.google.com/drive/folders/1vuI-YwELv8ZMxgxF7ioWN9ceHjArwV91) to run through the module. 

The dockerized package:
https://hub.docker.com/r/chinmaychinara/auto-resection-mask

## Steps to run
1. Make sure you have Docker Desktop installed and running in your machine. For more details check: https://hub.docker.com/
- For Apple Silicon, under `Settings > General > Virtual Machine Options > Choose Virtual Machine Manager (VMM)` set it to `Docker VMM` (available for  Docker Desktop >= 4.35) for better performance.
- If `Docker VMM` is not available then set the VMM as `Apple Virtualization framework` and check the `Use Rosetta for x86_64/amd64 emulation for Apple Silicon` and `VirtioFS` options.

3. <ins>Pull the image</ins>
- `docker pull chinmaychinara/auto-resection-mask:latest`

3. <ins>Run the image</ins>
- For Windows, Linux (with NVIDIA GPU):<br>`docker run --gpus all /path/to/your/data:/data chinmaychinara/auto-resection-mask:latest /data/preop.nii.gz /data/postop.nii.gz`<br>
  <ins>NOTE1:</ins> For GPU usage i.e. for the flag `gpus –all` to work configure [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) in your machine.<br>
  <ins>NOTE2:</ins> The image expects that your machine machine has CUDA 13.0 i.e. the NVIDIA driver >= 580.<br>
- For Windows, Linux, Apple Intel/Silicon (without NVIDIA GPU):<br>`docker run -v /path/to/your/data:/data chinmaychinara/auto-resection-mask:latest /data/preop.nii.gz /data/postop.nii.gz`

4. The outputs will be saved in the same folder as the input data if everything runs successfully. For details on how to interpret the results please refer to the [main repo](https://github.com/ajoshiusc/auto_resection_mask).
