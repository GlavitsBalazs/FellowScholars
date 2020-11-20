Setting up the training environment:
0. Have a fresh install of Ubuntu 20.04 Minimal. Open port 80 and 443 on your firewall. Have a `jupyter.@` DNS record for this machine.
1. Run `prepare.sh` as root.
2. Run `sudo usermod -aG docker $(whoami)` to access docker.
3. Reboot (required by the Nvidia driver).
4. Get the dataset with `nohup ./getdataset.sh &` (warning: takes at least 8 hours.)
5. Fill in the placeholders in `docker-compose.yml`.
6. Once the dataset has downloaded, `docker-compose up`.