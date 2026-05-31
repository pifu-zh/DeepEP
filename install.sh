#!/bin/bash

DISABLE_SM90_FEATURES=1 NVSHMEM_DIR=/usr/local/nvshmem  python3 setup.py install 2>&1|tee compile.log