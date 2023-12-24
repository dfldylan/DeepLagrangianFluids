# Lagrangian Fluid Simulation with Continuous Convolutions

![PyTorch badge](https://img.shields.io/badge/PyTorch-supported-brightgreen?style=flat&logo=pytorch)
![TensorFlow badge](https://img.shields.io/badge/TensorFlow-supported-brightgreen?style=flat&logo=tensorflow)

This repository contains code for our ICLR 2020 paper. 
We show how to train particle-based fluid simulation networks as CNNs using 
continuous convolutions. The code allows you to generate data, train your own 
model or just run a pretrained model.

<p align="center"> <img src="images/canyon.gif" alt="canyon video"> </p>

Please cite our paper [(pdf)](https://openreview.net/pdf?id=B1lDoJSYDH) if you find this code useful:
```
@inproceedings{Ummenhofer2020Lagrangian,
        title     = {Lagrangian Fluid Simulation with Continuous Convolutions},
        author    = {Benjamin Ummenhofer and Lukas Prantl and Nils Thuerey and Vladlen Koltun},
        booktitle = {International Conference on Learning Representations},
        year      = {2020},
}
```

To stay informed about updates we recommend to watch this repository.

## Environment

To ensure a consistent and easy-to-setup environment for running and developing this project, we provide a Docker-based solution. Docker allows you to encapsulate the project's environment and dependencies in a container, which can be run on any system that supports Docker, thereby avoiding common issues related to dependencies and platform-specific configurations.

### Setting Up the Docker Environment

1. **Building the Docker Image**: 
   Start by building a Docker image from the Dockerfile located in the root directory of the repository. This Dockerfile contains all the necessary instructions to create an environment that meets the requirements listed in the Dependencies section. Use the following command from the root of the repository to build the Docker image:

   ```bash
   docker build -t lagrangian-fluid-simulation .
   ```

   This command creates a Docker image named `lagrangian-fluid-simulation` based on the specifications in the Dockerfile.

2. **Starting the Docker Container**:
   Once the image is built, you can start a container from this image. The container is an instance of the image, providing an isolated environment for running your project. To start the container, run:

   ```bash
   docker run -it --name lfs-container lagrangian-fluid-simulation
   ```

   This command starts a new container named `lfs-container` from the `lagrangian-fluid-simulation` image and provides an interactive terminal to work with.

3. **Installing Additional Dependencies**:
   Inside the Docker container, you might need to install additional dependencies specific to your project or for further development. To do this, run the `deps.sh` script located in the repository's root directory:

   ```bash
   ./deps.sh
   ```

   Ensure that this script is executable (`chmod +x deps.sh` if necessary). This script should handle the installation of any additional dependencies that are not covered by the Dockerfile.

By following these steps, you will have a Docker container that provides a consistent and isolated environment for working with the Lagrangian Fluid Simulation project, as outlined in the Dependencies section.

## Dependencies

- Tensorflow 2.3 or PyTorch 1.6
- Open3D 0.11 or later with ML module (https://github.com/intel-isl/Open3D/)
- SPlisHSPlasH 2.4.0 (for generating training data and fluid particle sampling, https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)
- Tensorpack DataFlow (for reading data, ```pip install --upgrade git+https://github.com/tensorpack/dataflow.git```)
- python-prctl (needed by Tensorpack DataFlow; depends on libcap-dev, install with ```apt install libcap-dev``` )
- msgpack (```pip install msgpack``` )
- msgpack-numpy (```pip install msgpack-numpy```)
- python-zstandard (```pip install zstandard``` https://github.com/indygreg/python-zstandard)
- partio (https://github.com/wdas/partio)
- SciPy
- OpenVDB with python binding (optional for creating surface meshes, https://github.com/AcademySoftwareFoundation/openvdb)
- plyfile (optional for creating surface meshes, ```pip install plyfile```)

The versions match the configuration that we have tested on a system with Ubuntu 18.04.
SPlisHSPlasH 2.4.0 is required for generating training data (ensure that it is compiled in *Release* mode).
We recommend to use the latest versions for all other packages.


### Installing Open3D 0.11 and later with pip
The ML module is included in Open3D 0.11 and later and can simply be installed with
```bash
pip install open3d
```
Make sure that the version of your ML framework matches the version for which the ML ops in Open3D have been built.
For Open3D 0.11 this is CUDA 10.1, TensorFlow 2.3 and PyTorch 1.6.
If you cannot match this configuration it is recommended to build Open3D from source.


### Building Open3D with ML module from source.
At the moment Open3D needs to be build from source to make the code in this 
repo work. To build Open3D with the ML ops for Tensorflow and PyTorch do the 
following
```bash
git clone --recursive https://github.com/intel-isl/Open3D.git
# check the file Open3D/util/scripts/install-deps-ubuntu.sh
# for dependencies and install them. For more instructions see the Open3D documentation

mkdir Open3D/build
cd Open3D/build

# This builds the ml ops for both TensorFlow and PyTorch.
# If you don't need both frameworks you can disable the one you don't need with OFF.
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TENSORFLOW_OPS=ON -DBUILD_PYTORCH_OPS=ON -DBUILD_CUDA_MODULE=ON -DGLIBCXX_USE_CXX11_ABI=OFF
make install-pip-package
```

## Running the pretrained model

The pretrained network weights are in ```scripts/pretrained_model_weights.h5``` for TensorFlow and in ```scripts/pretrained_model_weights.pt``` for PyTorch.
The following code runs the network on the example scene
```bash
cd scripts
# with TensorFlow
./run_network.py --weights pretrained_model_weights.h5 \
                 --scene example_scene.json \
                 --output example_out \
                 --write-ply \
                 train_network_tf.py
# or with PyTorch
./run_network.py --weights pretrained_model_weights.pt \
                 --scene example_scene.json \
                 --output example_out \
                 --write-ply \
                 train_network_torch.py
```
The script writes point clouds with the particle positions as .ply files, which can be visualized with Open3D.
Note that SPlisHSPlasH is required for sampling the initial fluid volumes from ```.obj``` files.


## Training the network

### Data generation
The data generation scripts are in the ```datasets``` subfolder.
To generate the training and validation data 
 1. Set the path to the ```DynamicBoundarySimulator``` of SPlisHSPlasH in the ```datasets/splishsplash_config.py``` script.
 2. Run the script from within the datasets folder 
    ```bash
    cd datasets
    ./create_data.sh
    ```

### Data download
If you want to skip the data generation step you can download training and validation data from the links below.

| default data  | 34GB | [link](https://drive.google.com/file/d/1b3OjeXnsvwUAeUq2Z0lcrX7j9U7zLO07) |
|---------------|------|---------------------------------------------------------------------------|
| DPI dam break | 24GB | [link](https://drive.google.com/file/d/1_-aAd_GHX8StyKWZLpvSWeGQ3vyytf7L) |
| 6k box data   | 23GB | [link](https://drive.google.com/file/d/1eDFJD-wiTxzDzywSvXLgzffI25su1S1q) |

For the default data the training set has been generated with the scripts in this
repository and the validation data corresponds to the data used in the paper.

The DPI dam break data has been generated with the code from the DPI-Nets repo.
Note that the data has been scaled to match the particle radius used for our method.
See the ```scripts/dambreak.yaml``` config file for more information on the scale factor.

The 6k box data is a simplified version of the default data with a constant number
of particles and always uses a simple box as environment.



### Training scripts
To train the model with the generated data simply run one of the ```train_network_x.py``` scripts from within the ```scripts``` folder. 
```bash
cd scripts
# TensorFlow version
./train_network_tf.py default.yaml
# PyTorch version
./train_network_torch.py default.yaml
```
The scripts will create a folder ```train_network_tf_default``` or ```train_network_torch_default``` respectively with snapshots and log files.
The log files can be viewed with Tensorboard.

### Evaluating the network
To evaluate the network run the ```scripts/evaluate_network.py``` script like this
```bash
./evaluate_network.py --trainscript train_network_tf.py --cfg default.yaml
# or
./evaluate_network.py --trainscript train_network_torch.py --cfg default.yaml
```

This will create the file ```train_network_{tf,torch}_default_eval_50000.json```, which contains the 
individual errors between frame pairs.

The script will also print the overall errors. The output should look like 
this if you use the generated the data:
```{'err_n1': 0.000859004137852537, 'err_n2': 0.0024183266885233934, 'whole_seq_err': 0.030323669719872864}```

Note that the numbers differ from the numbers in the paper due to changes in 
the data generation:
 - We use Open3D to sample surface points to avoid shipping a modified 
   SPlisHSPlasH
 - The sequence of pseudorandom numbers used in the data generation is 
   different, which results in different scenes for training and testing.

If you have downloaded the validation data then the output should be similar to the numbers in the paper.
```{'err_n1': 0.000665973493194656, 'err_n2': 0.0018649007299291042, 'whole_seq_err': 0.03081335372162257}```

## Rendering

See the [scenes](scenes/README.md) directory for instructions on how to create and render the example scenes like the canyon.

## Licenses

Code and scripts are under the MIT license.

Data files in ```datasets/models``` and ```scripts/pretrained_model_weights.{h5,pt}``` are under the CDLA-Permissive-1.0 license.
