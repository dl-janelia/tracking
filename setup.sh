#!/usr/bin/env -S bash -i

if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "Hey, you should source this script, not execute it!"
    echo "Try: source setup.sh"
    echo "Exiting..."
    exit 1
fi

# Create environment
conda create -y -n 08-tracking python=3.11

# Activate environment
conda activate 08-tracking

# Install additional requirements
if [[ "$CONDA_DEFAULT_ENV" == "08-tracking" ]]; then
    echo "Environment activated successfully for package installs"
    pip install numpy "motile>=0.3" "traccuracy>=0.1.1" "geff>=1.1.3" "trackastra" "motile-tracker>=4.6,<5" "funtracks>=1.8,<2" "zarr<3" matplotlib ipywidgets nbformat pandas ipykernel
    python -m ipykernel install --user --name "08-tracking"
else
    echo "Failed to activate environment for package installs. Dependencies not installed!"
fi

conda deactivate

# Download data from s3
wget https://dl-at-mbl-data.s3.us-east-2.amazonaws.com/2025/09_tracking/data.zip
unzip data.zip
rm data.zip

# Alternatively, use the aws cli
# mkdir data
# aws s3 cp s3://dl-at-mbl-data/2025/09_tracking/ data/ --recursive --no-sign-request
