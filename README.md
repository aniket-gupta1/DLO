# Deep Lidar SLAM

## Setup and Installation
The code was tested on Ubuntu 20 with Cuda 11.3 and Python 3.10

Clone the repo
```
git clone https://github.com/aniket-gupta1/DLO.git --recursive
cd DLO
```

Create and activate the conda environment
```
conda env create -f environment.yml
conda activate DLO
```

Install torch_points_kernels separately. 
```
python3 -m pip install git+https://github.com/aniket-gupta1/torch-points.git
```

## Train the model
```
python main.py
```
