#!/bin/bash
# Create the two small Python environments used by the OSCAR RECO study.

set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
ENV_BASE=${ENV_BASE:-/oscar/scratch/$USER/mucoll/envs}
STORE_ENV="$ENV_BASE/reco-store"
TRAIN_SITE="$ENV_BASE/reco-train-site-v1"

mkdir -p "$ENV_BASE"

echo "Creating ROOT-to-HDF5 environment: $STORE_ENV"
python3 -m venv "$STORE_ENV"
"$STORE_ENV/bin/python" -m pip install --upgrade pip
"$STORE_ENV/bin/python" -m pip install \
    -r "$REPO/requirements-reco-store.txt"
"$STORE_ENV/bin/python" -c \
    'import awkward,h5py,numpy,uproot; print("store environment OK")'

echo "Creating TensorFlow-container extension: $TRAIN_SITE"
module load ngc-tensorflow-container/25.02-tf2-py3-j4zj
mkdir -p "$TRAIN_SITE"
apptainer exec --bind /oscar:/oscar "$NGC_TENSORFLOW_CONTAINER" \
    python -m pip install --upgrade --target "$TRAIN_SITE" \
    -r "$REPO/requirements-reco-train.txt"
PYTHONPATH="$TRAIN_SITE" apptainer exec --bind /oscar:/oscar \
    "$NGC_TENSORFLOW_CONTAINER" python -c \
    'import energyflow,h5py,matplotlib,sklearn,tensorflow,tf_keras; print("training environment OK")'

echo "OSCAR RECO environments are ready."
