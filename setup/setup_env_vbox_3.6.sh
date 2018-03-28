#!/usr/bin/env bash

# Must run this script from Vagrant /vagrant
# Tested on a Mac with Vagrant

echo "This will set up a python virtual environment inside Vagrant with python 3.6."
echo "This script must be run from inside Vagrant."
echo "This script requires sudo privileges and apt-get."

sleep 5

set -e # quits at first error

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3.6
sudo apt-get install python3-pip
sudo pip3 install virtualenv
cd ~
virtualenv -p /usr/bin/python3.6 env_python3.6
cd /vagrant

echo ""
echo "###################################"
echo ""
echo "Finished installing"