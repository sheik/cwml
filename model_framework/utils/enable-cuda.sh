#!/bin/bash
# script for enabling CUDA on ec2

# tensorflow 2.4 is built against cuda 11.0
yum install cuda-11-0

# enable persistenced, GPU will not operate with out this
sudo systemctl enable nvidia-persistenced
sudo systemctl start nvidia-persistenced

sudo systemctl enable nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager
