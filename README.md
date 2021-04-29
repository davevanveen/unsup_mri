# Unsupervised MRI

<br>
MRI recon using unsupervised neural networks

***
### List of contents
* [Setup and installation](#Setup-and-installation) <br>
* [Dataset](#Dataset) <br>
* [Running the code](#Running-the-code) <br>
* [References](#References) <br>
* [License](#License)
***

## Setup and installation

#### OS requirements
The code has been tested on Linux: Ubuntu 16.04.5

#### Python dependencies
The following packages are required to reproduce the experiments. Assuming the experiment is being performed in a docker container or a linux machine, the following libraries and packages need to be installed.

        apt-get update
        apt-get install python3.8
	pip install -r requirements.txt

Also install pytorch [here](https://pytorch.org/) according to your system specifications. If pip does not come automatically with your version of python, install manually [here](https://ehmatthes.github.io/pcc/chapter_12/installing_pip.html).

## Datasets
Experiments are performed on either the 2D [FastMRI](https://fastmri.org/dataset) dataset or an internal 3D MRI dataset. We note this reconstruction process can be applied on any image dataset, although the MRI-specific processing would need to be changed.

## Running the code
You may simply clone this repository and run the script `run_fastmri.py` to reproduce the results. **Note** that you need to download the [FastMRI](https://fastmri.org/dataset) dataset and change the **data path** (when loading the measurements) in each notebook accordingly, provided that you intend to run the code for MRI data (for MRI data, all of our experiments are performed on the validation sets--either single-coil or multi-coil).

## References
''Can Un-trained Neural Networks Compete with Trained Neural Networks at Image Reconstruction?,'' by Mohammad Zalbagi Darestani and Reinhard Heckel

## License
This project is covered by **Apache 2.0 License**.
