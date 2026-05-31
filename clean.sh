#!/bin/bash

NVSHMEM_DIR=/usr/local/nvshmem python3 setup.py clean

rm -rf build build_logs configure.log
rm -rf deep_ep.egg-info
rm -rf deep_ep/deep_ep_cpp.*.so
rm -rf dist