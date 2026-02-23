#!/bin/bash
# Wrapper script to run Julia with CUDA library path fix
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | grep -v "/usr/local/cuda-13/lib64" | paste -sd ":" -)
julia "$@"

