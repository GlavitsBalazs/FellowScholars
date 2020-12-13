# Training environment

We set up [Jupyter](https://github.com/jupyter/docker-stacks/tree/master/scipy-notebook) running on a VPS equipped with Nvidia GPUs (rented from Google Cloud) as our work environment.

To replicate, follow these steps on a fresh install of Ubuntu 20.04 Minimal.
1. Ensure that the machine is accesible from the internet. Open port 80 and 443 on your firewall. Install a `jupyter.@` DNS record.
2. Download all files from this directory into the home directory.
3. Run `prepare.sh` as root.
4. Run `sudo usermod -aG docker $(whoami)` to access docker.
5. Reboot (required by the Nvidia driver).
6. Get the dataset with `nohup ./getdataset.sh &` (Warning: takes at least 8 hours.)
7. While the dataset is downloading, edit `docker-compose.yml` for your needs. Fill in the placeholders.
8. Once the dataset has downloaded, `docker-compose up` will start Jupyter.
