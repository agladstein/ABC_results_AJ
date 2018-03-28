# ABC_results_AJ
ABC results for "Substructured population growth in the Ashkenazi Jews inferred with Approximate Bayesian Computation"

## Install and Environment Set up

### Virtual environment

#### Virtual Machine for non-Linux
If you are running on a non-Linux OS, we recommend using the virtual machine, Vagrant (can be used on Mac or PC). In order to run Vagrant, you will also need VirtualBox.

Download Vagrant from https://www.vagrantup.com/downloads.html

Download VirtualBox from https://www.virtualbox.org/

cd to the directory you want to work in and then download the repository,

```bash
git clone https://github.com/agladstein/ABC_results_AJ.git
```

Start Vagrant, ssh into Vagrant.
```bash
vagrant up
vagrant ssh
cd /vagrant
```

Install the virtual environment and install the requirements.
```bash
./setup/setup_env_vbox_3.6.sh
```