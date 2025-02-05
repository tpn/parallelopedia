#!/bin/bash

# Get the full path of this file.
ROOT=$(cd $(dirname $0); pwd)
# Add $ROOT/src/parallelopedia to PYTHONPATH.
PYTHONPATH=$ROOT/src:$PYTHONPATH python -Xgil=0 -m parallelopedia.server --ip 0.0.0.0 --port 4444 --threads 40 --log-level DEBUG --debug
