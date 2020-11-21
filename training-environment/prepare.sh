#!/bin/sh

# Run this script on a fresh install of Ubuntu 20.04 Minimal as root.

# Ubuntu wants to run GNOME, don't let it:
systemctl isolate multi-user.target && systemctl set-default multi-user.target

# Create the notebooks directory in advance and set its owner.
mkdir notebooks
chown ubuntu:users notebooks

apt-get update && apt-get upgrade -y
apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common unzip

# Install Nvidia driver, Docker and Nvida-Docker.
# The required version 455.45.01 of the Nvidia driver is retrieved from the CUDA repo.

curl -s -L https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

apt-get update
apt-get -y install nvidia-headless-455=455.45.01-0ubuntu1 nvidia-utils-455=455.45.01-0ubuntu1 \
docker-ce docker-ce-cli containerd.io docker-compose nvidia-container-toolkit nvidia-container-runtime nvidia-docker2
