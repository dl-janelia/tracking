#!/usr/bin/env -S bash -i

if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "Hey, you should source this script, not execute it!"
    echo "Try: source setup.sh"
    echo "Exiting..."
    exit 1
fi

# Create environment
uv sync

# Activate environment
source .venv/bin/activate

# Download data from s3
wget https://dl-at-mbl-data.s3.us-east-2.amazonaws.com/2026/tracking/data.zip
unzip data.zip
rm data.zip

# Alternatively, use the aws cli
# mkdir data
# aws s3 cp s3://dl-at-mbl-data/2025/09_tracking/ data/ --recursive --no-sign-request
