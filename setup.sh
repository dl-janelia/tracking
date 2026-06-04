#!/usr/bin/env -S bash -i

if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "Hey, you should source this script, not execute it!"
    echo "Try: source setup.sh"
    echo "Exiting..."
    exit 1
fi

# Create environment
conda create -n tracking python=3.12

# Activate environment
conda activate tracking

# Install dependencies
pip install "motile>=0.3" "traccuracy>=0.1.1" "geff>=1.1.3" "motile-tracker[all]>=4.6,<5" "funtracks>=1.8,<2" "zarr<3" numpy trackastra matplotlib ipywidgets nbformat pandas ipykernel jupyterlab

# Register the kernel under the name the notebooks expect
python -m ipykernel install --user --name tracking --display-name tracking

# Download data from s3
wget https://dl-at-mbl-data.s3.us-east-2.amazonaws.com/2026/tracking/data.zip
unzip data.zip
rm data.zip

# Alternatively, use the aws cli
# mkdir data
# aws s3 cp s3://dl-at-mbl-data/2026/tracking/ data/ --recursive --no-sign-request
