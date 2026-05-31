#!/bin/bash

export NVSHMEM_P2P_DISABLE=1

python3 tests/test_low_latency.py 2>&1|tee runtime.log
