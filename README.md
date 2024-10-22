# Unsupervised MRI

<br>
MRI recon using unsupervised neural networks

***
### List of contents
* [Setup](#Setup) <br>
* [Datasets](#Datasets) <br>
* [Demo](#Demo) <br>
* [References](#References) <br>
* [License](#License)
***

## Setup

The following packages are required to reproduce the experiments. Assuming the experiment is being performed in a docker container or a linux machine, the following libraries and packages need to be installed.

		apt-get update
		apt-get install python3.8
		pip install -r requirements.txt

Also install pytorch [here](https://pytorch.org/) according to your system specifications. If pip does not come automatically with your version of python, install manually [here](https://ehmatthes.github.io/pcc/chapter_12/installing_pip.html).

This was tested on Linux: Ubuntu 16.04.5

## Datasets
Experiments are performed on either the 2D [FastMRI](https://fastmri.org/dataset) dataset or an internal 3D MRI dataset. We note this reconstruction process can be applied on any image dataset, although the MRI-specific processing would need to be changed.

## Demo
See `demo/demo.ipynb` to run a simplified example. Additional functionality will be added shortly. 

## References
''Can Un-trained Neural Networks Compete with Trained Neural Networks at Image Reconstruction?,'' by Mohammad Zalbagi Darestani and Reinhard Heckel

## License
This project is covered by **Apache 2.0 License**.
