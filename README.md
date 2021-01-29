# ConvDecoder

<br>
Unsupervised MRI reconstruction based on the following paper:

**''Can Un-trained Neural Networks Compete with Trained Neural Networks at Image Reconstruction?,''** by Mohammad Zalbagi Darestani and Reinhard Heckel
***

### List of contents
* [Setup and installation](#Setup-and-installation) <br>
* [Dataset](#Dataset) <br>
* [Running the code](#Running-the-code) <br>
* [References](#References) <br>
* [License](#License)
***

# Setup and installation
On a normal computer, it takes aproximately 10 minutes to install all the required softwares and packages.

### OS requirements
The code has been tested on the following operating system:

	Linux: Ubuntu 16.04.5

### Python dependencies
To reproduce the results by running each of the jupyter notebooks, the following softwares are required. Assuming the experiment is being performed in a docker container or a linux machine, the following libraries and packages need to be installed.

        apt-get update
        apt-get install python3.6     # --> or any other system-specific command for installing python3 on your system.
		pip install jupyter
		pip install numpy
		pip install matplotlib
		pip install sigpy
		pip install h5py
		pip install scikit-image
		pip install runstats
		pip install pytorch_msssim
		pip install pytorch_lightning
		pip install test-tube
		pip install Pillow
		
If pip does not come with the version of python you installed, install pip manually from [here](https://ehmatthes.github.io/pcc/chapter_12/installing_pip.html). Also, install pytorch from [here](https://pytorch.org/) according to your system specifications. 

# Dataset
All the experiments are performed on the [FastMRI](https://fastmri.org/dataset) dataset--except the experiment for one set of experiments run on a separate internal (not publically available) MRI dataset

# Running the code
You may simply clone this repository and run the script `run_fastmri_expmt.py` to reproduce the results. **Note** that you need to download the [FastMRI](https://fastmri.org/dataset) dataset and change the **data path** (when loading the measurements) in each notebook accordingly, provided that you intend to run the code for MRI data (for MRI data, all of our experiments are performed on the validation sets--either single-coil or multi-coil).

# License
This project is covered by **Apache 2.0 License**.
