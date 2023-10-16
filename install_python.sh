#!/bin/bash
echo "Installing Python libraries..."
pip install numpy
pip install scipy==1.7.2
pip install pyyaml
pip install -U rospkg
pip install empy
sudo apt install ros-noetic-moveit
