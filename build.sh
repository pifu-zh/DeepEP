#!/bin/bash

NVSHMEM_DIR=/usr/local/nvshmem python3 setup.py build 2>&1|tee install.log