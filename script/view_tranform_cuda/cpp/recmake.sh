#!/bin/bash

if [[ $1 == "run" ]]; then
  echo run
else

if [[ $1 == "rm" || ! -d build ]]; then
    rm -rf build && mkdir build && cd build && ../../../cmake-3.25.0-linux-x86_64/bin/cmake .. && make -j2 && cd ..
else
    cd build && make -j2 && cd ../
fi

fi

# VT_RES_DIR=../

# ./build/bin/ViewTransformer.run $VT_RES_DIR/input $VT_RES_DIR/ground_truth
