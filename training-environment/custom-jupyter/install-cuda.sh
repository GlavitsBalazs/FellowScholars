#!/bin/sh

# This script installs the CUDA packages that tensorflow 2.4.0rc2 depends on.

apt-get update
apt-get install -yq software-properties-common gnupg
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
apt-get update
apt-get install -yq libcudnn8=8.0.5.39-1+cuda11.0 libcudnn8-dev=8.0.5.39-1+cuda11.0 cuda-toolkit-11-0 \
 cuda-11-0 cuda-cudart-11-0 cuda-cudart-dev-11-0 libcusparse-11-0 libcusparse-dev-11-0 libcublas-11-0 libcublas-dev-11-0
apt-get clean
rm -rf /var/lib/apt/lists/*