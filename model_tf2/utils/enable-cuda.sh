#!/bin/bash
# script for enabling CUDA on ec2

sudo systemctl enable nvidia-persistenced
sudo systemctl start nvidia-persistenced

sudo systemctl enable nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager
