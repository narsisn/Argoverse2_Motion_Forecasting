#!/bin/bash

# help url : https://github.com/argoai/av2-api/blob/main/DOWNLOAD.md
# Folder structure
mkdir -p dataset/argoverse
cd dataset/argoverse

# Settings
export INSTALL_DIR=$HOME/.local/bin
export PATH=$PATH:$INSTALL_DIR
export S5CMD_URI=https://github.com/peak/s5cmd/releases/download/v1.4.0/s5cmd_1.4.0_$(uname | sed 's/Darwin/macOS/g')-64bit.tar.gz

mkdir -p $INSTALL_DIR
curl -sL $S5CMD_URI | tar -C $INSTALL_DIR -xvzf - s5cmd

# Download about 59G data test:4.3G , train:49G, val: 6.1G
s5cmd --no-sign-request cp s3://argoai-argoverse/av2/motion-forecasting/*

