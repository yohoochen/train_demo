#!/bin/bash
# usage: run.sh camera or run.sh video
set -e

cd build
rm -rf *
cmake ..
make -j12
cd ../
./bin/buildEngine
#./bin/runDet
./bin/pose_test
